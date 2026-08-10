// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package compaction summarizes older session events so an agent's prompt stays
// small as its conversation grows.
//
// A compaction never modifies or deletes history. Summarizing a range of events
// appends one new [session.Event] carrying a [session.EventCompaction] that
// records the covered timestamp range and the summary content. When the next
// prompt is built, the raw events inside that range are dropped and the summary
// is materialized in their place.
//
// # What each strategy achieves
//
// Sliding window replaces each group of invocations with one summary, but
// summaries are never themselves re-summarized. Prompt size therefore still
// grows with conversation length, at a reduced constant factor rather than
// being bounded.
//
// Tail retention is what bounds it: each new summary is seeded with the
// previous one, so history stays as a single rolling summary plus a raw tail.
// An agent that needs a genuine ceiling on prompt size should enable it, either
// on its own or alongside the sliding window.
//
// Compaction is enabled per runner. See the EventsCompactionConfig field on
// runner.Config:
//
//	r, err := runner.New(runner.Config{
//		AppName:        "my-app",
//		Agent:          rootAgent,
//		SessionService: session.InMemoryService(),
//		EventsCompactionConfig: &compaction.Config{
//			CompactionInterval: 3,
//			OverlapSize:        1,
//		},
//	})
package compaction

import (
	"context"
	"errors"
	"fmt"

	"google.golang.org/adk/v2/session"
)

// ErrCompaction marks an error as a compaction failure rather than a failure of
// the turn itself.
//
// Compaction is bookkeeping: the events of the turn are already persisted
// before it runs, so a failure costs a smaller prompt later, not the user's
// answer. It still surfaces, because a summarizer that never succeeds is worth
// knowing about, but a caller that would rather log it than fail the turn can
// tell the two apart:
//
//	for event, err := range r.Run(...) {
//		if errors.Is(err, compaction.ErrCompaction) {
//			log.Printf("compaction failed: %v", err)
//			continue
//		}
//		...
//	}
var ErrCompaction = errors.New("context compaction failed")

// Config configures context compaction for an application.
//
// Two independent strategies are available, and at least one must be enabled.
// A Config that enables neither is rejected by [Config.Validate], because it
// would cost a configuration step and do nothing; leave the whole Config nil to
// disable compaction:
//
//   - Sliding window (CompactionInterval, OverlapSize) runs after an invocation
//     completes and summarizes whole invocations at a time.
//   - Tail retention (TokenThreshold, EventRetentionSize) runs inside an
//     invocation before a model call and summarizes everything but the most
//     recent events once the prompt grows past a token budget.
type Config struct {
	// CompactionInterval is the number of new user-initiated invocations that,
	// once fully represented in the session's events, triggers a sliding-window
	// compaction. Zero, the default, disables sliding-window compaction.
	//
	// It also bounds the window: one compaction covers at most this many new
	// invocations, so enabling compaction on a session that already has a long
	// history drains the backlog a window at a time rather than summarizing all
	// of it in one call.
	CompactionInterval int

	// OverlapSize is how many already-compacted invocations to pull back into
	// the next sliding window, creating an overlap between consecutive
	// summaries for continuity. Only meaningful alongside CompactionInterval.
	//
	// The overlap is repeated, not shared: an invocation pulled back in is
	// described by both summaries, so the model sees it twice and the prompt
	// carries roughly OverlapSize invocations of extra text per summary. That
	// is the cost of the continuity, and it cannot be trimmed away afterwards,
	// because by then the repetition lives inside summary prose rather than in
	// the ranges. Leave it at zero unless summaries are visibly losing the
	// thread between windows.
	OverlapSize int

	// TokenThreshold is the prompt token count at which intra-invocation
	// tail-retention compaction fires before a model call. Zero, the default,
	// disables tail-retention compaction.
	TokenThreshold int

	// EventRetentionSize is how many of the most recent events are kept raw
	// when tail-retention compaction fires. Everything older is summarized.
	// Only meaningful alongside TokenThreshold, and required with it: at zero
	// the window would extend to the newest event, which includes the question
	// the model is about to answer, so the turn in progress would be summarized
	// out of its own prompt.
	EventRetentionSize int

	// Summarizer produces the summary content. When nil, the runner supplies an
	// [LLMSummarizer] backed by the root agent's model, which therefore has to
	// be an LLM agent.
	Summarizer Summarizer
}

// hasSlidingWindow reports whether sliding-window compaction is enabled.
func (c *Config) hasSlidingWindow() bool {
	return c != nil && c.CompactionInterval > 0
}

// hasTailRetention reports whether tail-retention compaction is enabled.
func (c *Config) hasTailRetention() bool {
	return c != nil && c.TokenThreshold > 0
}

// Validate reports whether the configuration is usable.
//
// A nil Config is valid and means compaction is disabled. A non-nil Config with
// no strategy enabled is not: allocating one and setting nothing is a mistake
// worth reporting rather than silently doing nothing, and nil already expresses
// "disabled".
func (c *Config) Validate() error {
	if c == nil {
		return nil
	}
	if c.CompactionInterval < 0 {
		return fmt.Errorf("CompactionInterval must not be negative, got %d", c.CompactionInterval)
	}
	if c.OverlapSize < 0 {
		return fmt.Errorf("OverlapSize must not be negative, got %d", c.OverlapSize)
	}
	if c.TokenThreshold < 0 {
		return fmt.Errorf("TokenThreshold must not be negative, got %d", c.TokenThreshold)
	}
	if c.EventRetentionSize < 0 {
		return fmt.Errorf("EventRetentionSize must not be negative, got %d", c.EventRetentionSize)
	}
	if c.OverlapSize > 0 && c.CompactionInterval == 0 {
		return fmt.Errorf("OverlapSize is set to %d but CompactionInterval is 0, so sliding-window compaction never runs", c.OverlapSize)
	}
	if c.TokenThreshold > 0 && c.EventRetentionSize == 0 {
		return fmt.Errorf("TokenThreshold is set to %d but EventRetentionSize is 0, so a compaction would summarize the whole conversation including the turn being answered", c.TokenThreshold)
	}
	if c.EventRetentionSize > 0 && c.TokenThreshold == 0 {
		return fmt.Errorf("EventRetentionSize is set to %d but TokenThreshold is 0, so tail-retention compaction never runs", c.EventRetentionSize)
	}
	if !c.hasSlidingWindow() && !c.hasTailRetention() {
		return fmt.Errorf("no compaction strategy is enabled, set CompactionInterval or TokenThreshold (or leave the whole config nil to disable compaction)")
	}
	return nil
}

// Summarizer compacts a range of events into a single summary event.
//
// Implement it to control which parts of an event reach the summary and how the
// summary is produced; [LLMSummarizer] is the default implementation.
type Summarizer interface {
	// SummarizeEvents summarizes events into one new event carrying the result
	// on its Actions.Compaction field. It returns a nil event when no summary
	// was produced, which callers treat as "skip this compaction" rather than
	// as an error. The events passed in are never modified.
	SummarizeEvents(ctx context.Context, events []*session.Event) (*session.Event, error)
}

// IsCompactionEvent reports whether ev carries a context-compaction summary
// that can actually be shown to a model: it declares a compaction, and that
// compaction has content.
//
// Use it to count stored summaries, or to decide what to materialize into a
// prompt. Note that it answers "is there a usable summary here", not "is this
// event bookkeeping rather than conversation" — an event whose compaction has
// no content is still bookkeeping, and this returns false for it. Only
// [session.EventActions.Compaction] being non-nil answers the second question.
func IsCompactionEvent(ev *session.Event) bool {
	return ev != nil && ev.Actions.Compaction != nil && ev.Actions.Compaction.CompactedContent != nil
}
