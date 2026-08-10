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

package compactioninternal

import (
	"context"
	"fmt"
	"slices"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// Apply rewrites an event list so compaction summaries stand in for the events
// they cover. It is what turns a stored compaction into a smaller prompt.
//
// Each surviving compaction event is replaced by a model-authored event holding
// its summary content, positioned at the compaction's end timestamp. Raw events
// falling inside a surviving range are dropped. A compaction whose range
// another compaction fully contains is discarded along with its summary, so
// re-summarized ranges do not appear twice.
//
// Finally, function calls that a summary swallowed but whose responses arrived
// later are restored, so call and response stay paired.
//
// events is not modified, and is returned unchanged when it holds no
// compactions.
func Apply(events []*session.Event) []*session.Event {
	if !slices.ContainsFunc(events, hasCompaction) {
		return events
	}
	return recoverCompactedFunctionCalls(substituteSummaries(events), events)
}

// hasCompaction reports whether ev declares a compaction at all, usable or not.
// Apply keys off this rather than [IsCompactionEvent] so that a malformed
// compaction is still stripped from the prompt instead of leaking through as a
// contentless raw event.
func hasCompaction(ev *session.Event) bool {
	return ev != nil && ev.Actions.Compaction != nil
}

// keptRange is a compaction range that survived subsumption, along with the
// stream position of the event that declared it.
type keptRange struct {
	index int
	rng   *session.EventCompaction
}

// substituteSummaries drops raw events covered by a surviving compaction and
// materializes each surviving summary in their place, preserving chronological
// order.
func substituteSummaries(events []*session.Event) []*session.Event {
	var kept []keptRange
	for i, ev := range events {
		if !compaction.IsCompactionEvent(ev) {
			continue
		}
		if ev.Actions.Compaction.EndTimestamp.Before(ev.Actions.Compaction.StartTimestamp) {
			// An inverted range covers nothing; materializing its summary would
			// duplicate content the raw events still supply. NewSummaryEvent
			// rejects these, but session.EventCompaction is a plain struct that
			// callers can also build directly.
			continue
		}
		if isCompactionSubsumed(i, ev.Actions.Compaction, events) {
			continue
		}
		kept = append(kept, keptRange{index: i, rng: ev.Actions.Compaction})
	}

	// Each surviving summary is emitted where the first event it covers sat,
	// and the events it covers are dropped.
	//
	// Stream position rather than timestamp: sorting the result on timestamp
	// could reorder raw events whose timestamps disagree with their arrival
	// order -- clock skew between writers, or the microsecond truncation the
	// SQL backend applies -- and so put a function response ahead of the call
	// it answers.
	//
	// The first covered event rather than the compaction event's own position:
	// a compaction event is appended after the range it covers, but not
	// necessarily right after it. Tail retention leaves a raw tail in between,
	// and emitting the summary where the compaction event sits would show the
	// model a summary of older history after the recent turns that follow it.
	summariesAt := make(map[int][]keptRange, len(kept))
	for _, k := range kept {
		at := summaryIndex(events, k)
		summariesAt[at] = append(summariesAt[at], k)
	}

	out := make([]*session.Event, 0, len(events))
	for i, ev := range events {
		for _, k := range summariesAt[i] {
			summary := *events[k.index]
			summary.Author = "model"
			summary.Timestamp = k.rng.EndTimestamp
			summary.LLMResponse.Content = k.rng.CompactedContent
			out = append(out, &summary)
		}
		if ev == nil {
			// A nil entry is not conversation and nothing can cover it.
			// Dropping it keeps Apply total over its input: it is reachable
			// from an exported entry point, so a malformed event list should
			// not panic deep inside coverage arithmetic.
			continue
		}
		if hasCompaction(ev) {
			// An event declaring a compaction is bookkeeping, never
			// conversation: its content slot holds nothing to show the model,
			// and its summary was emitted above at the position of the range it
			// covers.
			continue
		}
		if isCovered(i, ev, kept) {
			continue
		}
		out = append(out, ev)
	}
	return out
}

// summaryIndex is the stream position at which k's summary materializes: where
// the first event it covers sat, or the compaction event itself when it covers
// nothing left in the stream.
func summaryIndex(events []*session.Event, k keptRange) int {
	for i, ev := range events {
		if ev == nil || hasCompaction(ev) {
			continue
		}
		if coveredBy(i, ev, k) {
			return i
		}
	}
	return k.index
}

// isCovered reports whether the raw event at index i falls inside a surviving
// compaction range.
func isCovered(i int, ev *session.Event, kept []keptRange) bool {
	if ev == nil {
		return false
	}
	for _, k := range kept {
		if coveredBy(i, ev, k) {
			return true
		}
	}
	return false
}

// coveredBy reports whether the raw event at index i falls inside k's range.
// Only a compaction appearing later in the stream can cover an event: a summary
// never covers events recorded after it was written.
func coveredBy(i int, ev *session.Event, k keptRange) bool {
	if i >= k.index {
		return false
	}
	return !ev.Timestamp.Before(k.rng.StartTimestamp) && !ev.Timestamp.After(k.rng.EndTimestamp)
}

// recoverCompactedFunctionCalls re-injects function-call events that compaction
// removed but whose responses survived.
//
// The case this exists for is a paused long-running tool call: the call and its
// placeholder response are compacted together, then the real result arrives on
// resume as a later event that no summary covers. That surviving response would
// be orphaned, which breaks the call/response pairing prompt assembly requires.
//
// For each orphaned response the original call event is restored from
// sourceEvents (the pre-substitution list) and inserted just before the first
// surviving response referencing it. The whole call event comes back so
// parallel calls stay intact, and for every sibling call in it whose response
// was also compacted away, the freshest response is re-injected too, so the
// sibling does not surface as a phantom pending call.
//
// Only long-running calls are recovered, and that is the only shape this can
// legitimately arise in. longestSelfContainedPrefix guarantees the summarized
// window is balanced, so every call inside it had its response inside it too.
// The one way a response outlives its call is a second response for the same
// call ID arriving after the window, which is exactly the long-running pattern:
// a placeholder response closes the pair, the pair is compacted, and the real
// result lands later.
//
// An unmatched response with no long-running call is a genuine inconsistency,
// and it is left alone rather than guessed at. Recovering it would invent a call
// that never happened, hiding the underlying bug instead of exposing it.
//
// Be aware of where such a response ends up:
// rearrangeEventsForLatestFunctionResponse errors on it only when it is the
// final event, while rearrangeEventsForFunctionResponsesInHistory drops any
// response it cannot pair with a call. So a mid-history orphan disappears from
// the prompt silently rather than loudly. If that ever needs to be made loud,
// the fix belongs in those two functions, not here.
func recoverCompactedFunctionCalls(events, sourceEvents []*session.Event) []*session.Event {
	presentCalls := make(map[string]struct{})
	presentResponses := make(map[string]struct{})
	for _, ev := range events {
		for _, call := range utils.FunctionCalls(utils.Content(ev)) {
			presentCalls[call.ID] = struct{}{}
		}
		for _, resp := range utils.FunctionResponses(utils.Content(ev)) {
			presentResponses[resp.ID] = struct{}{}
		}
	}

	orphaned := make(map[string]struct{})
	for id := range presentResponses {
		if _, ok := presentCalls[id]; !ok && id != "" {
			orphaned[id] = struct{}{}
		}
	}
	if len(orphaned) == 0 {
		return events
	}

	// The long-running call events matching the orphaned responses.
	callEventByID := make(map[string]*session.Event)
	for _, ev := range sourceEvents {
		for _, call := range utils.FunctionCalls(utils.Content(ev)) {
			if _, ok := orphaned[call.ID]; !ok {
				continue
			}
			if _, ok := callEventByID[call.ID]; ok {
				continue
			}
			if slices.Contains(ev.LongRunningToolIDs, call.ID) {
				callEventByID[call.ID] = ev
			}
		}
	}
	if len(callEventByID) == 0 {
		return events
	}

	// Freshest response event per call ID, so a re-injected sibling carries its
	// final result rather than an intermediate placeholder.
	finalResponseByID := make(map[string]*session.Event)
	for _, ev := range sourceEvents {
		for _, resp := range utils.FunctionResponses(utils.Content(ev)) {
			if prev, ok := finalResponseByID[resp.ID]; !ok || !ev.Timestamp.Before(prev.Timestamp) {
				finalResponseByID[resp.ID] = ev
			}
		}
	}

	result := make([]*session.Event, 0, len(events)+len(callEventByID))
	reinjected := make(map[string]struct{})
	for _, ev := range events {
		for _, resp := range utils.FunctionResponses(utils.Content(ev)) {
			callEvent, ok := callEventByID[resp.ID]
			if !ok {
				continue
			}
			if _, done := reinjected[resp.ID]; done {
				continue
			}

			result = append(result, callEvent)

			// Every call in the recovered event is now present, including the
			// parallel siblings that came along for the ride.
			var siblings []*session.Event
			for _, call := range utils.FunctionCalls(utils.Content(callEvent)) {
				reinjected[call.ID] = struct{}{}
				if _, present := presentResponses[call.ID]; present {
					continue
				}
				if sibling, ok := finalResponseByID[call.ID]; ok && !slices.Contains(siblings, sibling) {
					siblings = append(siblings, sibling)
				}
			}
			result = append(result, siblings...)
		}
		result = append(result, ev)
	}
	return result
}

// RangeRaced reports whether the session gained an event inside summary's range
// while the summary was being produced.
//
// A summary records the span it covers as an inclusive timestamp range, and
// prompt assembly drops everything inside that range. Summarizing takes a model
// call, so a concurrent invocation on the same session can append inside the
// chosen span while it is in flight. Recording the summary anyway would drop
// those turns from every later prompt without ever having summarized them.
//
// selectedFrom is the session state the window was chosen from, and latest is a
// fresh read taken after summarizing. An event inside the range that is present
// in latest but absent from selectedFrom arrived too late to be summarized.
// Comparing the two states makes this exact rather than a guess about
// timestamps.
//
// Callers discard the summary when this returns true.
func RangeRaced(latest, selectedFrom session.Session, summary *session.Event) bool {
	rng := summary.Actions.Compaction
	if latest == nil || selectedFrom == nil || rng == nil {
		return false
	}

	known := make(map[string]struct{})
	for _, ev := range collect(selectedFrom) {
		known[ev.ID] = struct{}{}
	}

	for _, ev := range collect(latest) {
		if hasCompaction(ev) {
			continue
		}
		if ev.Timestamp.Before(rng.StartTimestamp) || ev.Timestamp.After(rng.EndTimestamp) {
			continue
		}
		if _, seen := known[ev.ID]; !seen {
			return true
		}
	}
	return false
}

// ReloadSession re-reads s from svc and returns the stored session.
//
// Compaction must not run against the session handle it was handed. That handle
// is a snapshot taken before the work started, so a concurrent invocation on the
// same session may have appended events it cannot see, and summarizing against
// it records a range covering those events without having summarized them. It
// may also be a wrapper that an agent installed over the real session, and every
// session service type-asserts on its own concrete type, so appending to a
// wrapper fails.
//
// Re-reading solves both: the result is current, and it is whatever concrete
// type the service issues.
func ReloadSession(ctx context.Context, svc session.Service, s session.Session) (session.Session, error) {
	if svc == nil || s == nil {
		return nil, fmt.Errorf("cannot re-read the session: no session service")
	}
	resp, err := svc.Get(ctx, &session.GetRequest{
		AppName:   s.AppName(),
		UserID:    s.UserID(),
		SessionID: s.ID(),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to re-read the session: %w", err)
	}
	if resp == nil || resp.Session == nil {
		return nil, fmt.Errorf("session %q disappeared while compacting", s.ID())
	}
	return resp.Session, nil
}

// sessionUnwrapper is implemented by a [session.Session] that decorates another
// one. Nothing in the public API exposes it: the decorators are unexported types
// that happen to carry the method.
type sessionUnwrapper interface {
	Unwrap() session.Session
}

// UnwrapSession returns the innermost session s decorates, or s itself.
//
// An agent may wrap the session it hands to a sub-agent so the sub-agent's
// prompt sees a synthetic first-turn seed. That wrapper is fine to read through
// but must not be compacted against: every session service type-asserts on its
// own concrete type, so appending to a wrapper fails outright, and the seed is
// not durable, so recording a range over it would cover an event no store holds.
//
// Unwrapping rather than re-reading is deliberate. It preserves object identity
// with the session the wrapper delegates to, so an event appended here is
// visible through the wrapper immediately. A freshly read session would be a
// different object, and the summary would not reach the prompt being assembled.
func UnwrapSession(s session.Session) session.Session {
	for {
		w, ok := s.(sessionUnwrapper)
		if !ok {
			return s
		}
		inner := w.Unwrap()
		if inner == nil {
			return s
		}
		s = inner
	}
}
