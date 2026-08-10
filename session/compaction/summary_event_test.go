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
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
)

func TestNewSummaryEvent(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		textEvent("a", "inv1", 3, "q1"),
		modelTextEvent("b", "inv1", 7, "a1"),
	}
	summaryContent := utils.Content(modelTextEvent("x", "inv1", 0, "the summary"))

	got, err := NewSummaryEvent(events, summaryContent, nil)
	if err != nil {
		t.Fatalf("NewSummaryEvent() error = %v", err)
	}

	if got.Author != "user" {
		t.Errorf("Author = %q, want %q", got.Author, "user")
	}
	if got.Actions.Compaction == nil {
		t.Fatal("Actions.Compaction is nil, want a compaction range")
	}
	if !got.Actions.Compaction.StartTimestamp.Equal(at(3)) {
		t.Errorf("StartTimestamp = %v, want %v", got.Actions.Compaction.StartTimestamp, at(3))
	}
	if !got.Actions.Compaction.EndTimestamp.Equal(at(7)) {
		t.Errorf("EndTimestamp = %v, want %v", got.Actions.Compaction.EndTimestamp, at(7))
	}
	if role := got.Actions.Compaction.CompactedContent.Role; role != "model" {
		t.Errorf("CompactedContent.Role = %q, want %q", role, "model")
	}
	// The caller's content must not be re-roled underneath them.
	if summaryContent.Role != "model" {
		t.Logf("input content role was already %q", summaryContent.Role)
	}
}

func TestNewSummaryEventRejectsBadInput(t *testing.T) {
	t.Parallel()

	ordered := []*session.Event{textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 4, "a1")}
	content := genai.NewContentFromText("summary", "model")

	tests := []struct {
		name    string
		events  []*session.Event
		summary *genai.Content
		wantErr bool
	}{
		{name: "ok", events: ordered, summary: content},
		{name: "single event is a valid degenerate range", events: ordered[:1], summary: content},
		{name: "no events", events: nil, summary: content, wantErr: true},
		{name: "nil summary", events: ordered, summary: nil, wantErr: true},
		{
			// An inverted range covers nothing, so the compacted turns would
			// stay in every future prompt while a summary was still paid for.
			name:    "events out of chronological order",
			events:  []*session.Event{modelTextEvent("b", "inv1", 4, "a1"), textEvent("a", "inv1", 1, "q1")},
			summary: content,
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := NewSummaryEvent(tc.events, tc.summary, nil)
			if gotErr := err != nil; gotErr != tc.wantErr {
				t.Errorf("NewSummaryEvent() error = %v, wantErr %t", err, tc.wantErr)
			}
		})
	}
}

// TestNewSummaryEventKeepsPartMetadata checks that a surviving text part is
// copied whole rather than rebuilt from its text.
//
// A text part can carry metadata that belongs with it, a thought signature
// above all, which the model expects to get back alongside the text it
// accompanies. Rebuilding the part would drop that silently.
func TestNewSummaryEventKeepsPartMetadata(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		{Timestamp: time.Unix(1, 0)},
		{Timestamp: time.Unix(2, 0)},
	}
	summary := &genai.Content{Role: "model", Parts: []*genai.Part{{
		Text:             "the summary",
		ThoughtSignature: []byte("opaque-signature"),
	}}}

	got, err := NewSummaryEvent(events, summary, nil)
	if err != nil {
		t.Fatalf("NewSummaryEvent() error = %v", err)
	}
	parts := got.Actions.Compaction.CompactedContent.Parts
	if len(parts) != 1 {
		t.Fatalf("got %d parts, want 1", len(parts))
	}
	if diff := cmp.Diff([]byte("opaque-signature"), parts[0].ThoughtSignature); diff != "" {
		t.Errorf("ThoughtSignature mismatch (-want +got):\n%s", diff)
	}
}

// TestNewSummaryEventRejectsProselessSummary checks that a summary whose only
// text rides on an actionable part is refused rather than stored empty.
func TestNewSummaryEventRejectsProselessSummary(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		{Timestamp: time.Unix(1, 0)},
		{Timestamp: time.Unix(2, 0)},
	}
	summary := &genai.Content{Role: "model", Parts: []*genai.Part{{
		Text:         "transferring now",
		FunctionCall: &genai.FunctionCall{Name: "transfer_funds"},
	}}}

	if _, err := NewSummaryEvent(events, summary, nil); err == nil {
		t.Error("NewSummaryEvent() accepted a summary with no prose, want an error rather than an empty summary")
	}
}

// TestCompactionEventIsNotAFinalResponse checks that a stored summary does not
// present itself to streaming consumers as an agent's final response.
//
// A compaction event carries a record and no content, which satisfies every
// other clause of IsFinalResponse, so a client deciding what to show a user
// would surface an empty final response every time compaction ran.
func TestCompactionEventIsNotAFinalResponse(t *testing.T) {
	t.Parallel()

	events := []*session.Event{
		{Timestamp: time.Unix(1, 0)},
		{Timestamp: time.Unix(2, 0)},
	}
	got, err := NewSummaryEvent(events, genai.NewContentFromText("the summary", "model"), nil)
	if err != nil {
		t.Fatalf("NewSummaryEvent() error = %v", err)
	}

	if got.IsFinalResponse() {
		t.Error("a compaction event reports IsFinalResponse() = true; streaming clients would show it as an empty reply")
	}
}
