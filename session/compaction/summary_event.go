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

package compaction

import (
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// NewSummaryEvent builds the event a [Summarizer] returns: an event carrying
// summary as an [session.EventCompaction] covering the range spanned by events.
//
// events must be non-empty and in chronological order, and summary must be
// non-nil; usage may be nil. An error is returned rather than a silently broken
// event, because an inverted range covers nothing and would leave the compacted
// turns in every future prompt while still consuming a summary.
//
// [session.EventCompaction] is a plain struct with no constructor to validate
// in, so the check lives here instead, at the supported way to build one.
func NewSummaryEvent(events []*session.Event, summary *genai.Content, usage *genai.GenerateContentResponseUsageMetadata) (*session.Event, error) {
	if len(events) == 0 {
		return nil, fmt.Errorf("cannot summarize an empty event list")
	}
	// An empty summary is rejected, not just a nil one. Recording a compaction
	// whose content says nothing deletes the covered turns from every future
	// prompt and puts nothing in their place, which is worse than not
	// compacting at all.
	if !hasText(summary) {
		return nil, fmt.Errorf("summary content is empty, so compacting would delete the covered events and replace them with nothing")
	}
	// NewSummaryEvent is exported and called by third-party Summarizer
	// implementations, so a nil element is an input to reject rather than a
	// panic to hand back.
	for i, ev := range events {
		if ev == nil {
			return nil, fmt.Errorf("events[%d] is nil", i)
		}
	}
	start, end := events[0].Timestamp, events[len(events)-1].Timestamp
	if end.Before(start) {
		return nil, fmt.Errorf("events are not in chronological order: first event is at %v, last at %v", start, end)
	}

	// Only prose survives into the stored summary. Whatever the summarizer
	// returns is injected into later prompts verbatim, so a non-text part
	// reaches the model as if the framework had produced it. A hallucinated or
	// maliciously supplied FunctionCall would arrive unpaired, and a model may
	// act on it. A summary is prose by definition, so anything else is dropped.
	//
	// A surviving part is copied whole rather than rebuilt from its text. A
	// text part can carry metadata that belongs with it and that the model
	// expects back, a thought signature above all, and rebuilding would drop
	// that silently.
	content := genai.Content{Role: "model"}
	for _, p := range summary.Parts {
		if !isProse(p) {
			continue
		}
		part := *p
		content.Parts = append(content.Parts, &part)
	}
	if len(content.Parts) == 0 {
		return nil, fmt.Errorf("summary content holds no prose, so compacting would delete the covered events and replace them with nothing")
	}

	// The summary inherits the branch and isolation scope of what it covers.
	// Without them it carries Branch "" and IsolationScope "", which every
	// branch filter admits and which makes it visible outside the scope its
	// source events belonged to, leaking scoped content across the boundary the
	// filters exist to enforce.
	branch, scope := events[0].Branch, events[0].IsolationScope

	return &session.Event{
		// Authored as "user" because a summary is injected context rather than
		// something the agent said. It is re-authored as "model" when
		// materialized into a prompt, so the model reads it as prior context.
		Author:         "user",
		Branch:         branch,
		IsolationScope: scope,
		Actions: session.EventActions{
			Compaction: &session.EventCompaction{
				StartTimestamp:   start,
				EndTimestamp:     end,
				CompactedContent: &content,
			},
		},
		LLMResponse: model.LLMResponse{UsageMetadata: usage},
	}, nil
}

// isProse reports whether p is plain text and nothing else.
//
// Exactly one field of a [genai.Part] is meant to be set, so a part that
// carries any of the actionable payloads is not prose whatever else is on it.
// Such a part is dropped rather than reduced to its text: the text is not what
// makes it dangerous, and dropping is the conservative half of the choice.
func isProse(p *genai.Part) bool {
	if p == nil || p.Text == "" {
		return false
	}
	return p.FunctionCall == nil &&
		p.FunctionResponse == nil &&
		p.ExecutableCode == nil &&
		p.CodeExecutionResult == nil &&
		p.FileData == nil &&
		p.InlineData == nil &&
		p.ToolCall == nil &&
		p.ToolResponse == nil
}
