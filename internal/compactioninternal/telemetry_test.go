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
	"errors"
	"testing"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/telemetry"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// spanRecorder installs an in-memory tracer for the calling test.
func spanRecorder(t *testing.T) *tracetest.InMemoryExporter {
	t.Helper()
	exp := tracetest.NewInMemoryExporter()
	telemetry.OverrideTracerForTesting(t, sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp)))
	return exp
}

// attrs flattens a span's attributes for lookup by key.
func attrs(kvs []attribute.KeyValue) map[string]attribute.Value {
	out := make(map[string]attribute.Value, len(kvs))
	for _, kv := range kvs {
		out[string(kv.Key)] = kv.Value
	}
	return out
}

func TestSlidingWindowEmitsSpan(t *testing.T) {
	exp := spanRecorder(t)

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
	}
	cfg := &compaction.Config{CompactionInterval: 2, OverlapSize: 1, Summarizer: &fakeSummarizer{summary: "sum"}}

	got, err := SlidingWindow(context.Background(), cfg, &staticSession{events: events})
	if err != nil {
		t.Fatalf("SlidingWindow() error = %v", err)
	}
	if got == nil {
		t.Fatal("SlidingWindow() produced no summary")
	}

	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	span := spans[0]
	if want := "compact_events sliding_window"; span.Name != want {
		t.Errorf("span name = %q, want %q", span.Name, want)
	}
	if span.Status.Code == codes.Error {
		t.Errorf("span status = error, want unset: %v", span.Status)
	}

	a := attrs(span.Attributes)
	for key, want := range map[string]string{
		"gen_ai.operation.name":             "compact_events",
		"gen_ai.conversation.id":            "sess",
		"gen_ai.compaction.trigger":         "sliding_window",
		"gen_ai.compaction.summarizer_type": "fakeSummarizer",
		"gen_ai.compaction.result_event_id": got.ID,
	} {
		if a[key].AsString() != want {
			t.Errorf("attribute %s = %q, want %q", key, a[key].AsString(), want)
		}
	}
	if a["gen_ai.compaction.event_count"].AsInt64() != 4 {
		t.Errorf("event_count = %d, want 4", a["gen_ai.compaction.event_count"].AsInt64())
	}
	if a["gen_ai.compaction.compaction_interval"].AsInt64() != 2 {
		t.Errorf("compaction_interval = %d, want 2", a["gen_ai.compaction.compaction_interval"].AsInt64())
	}
	if a["gen_ai.compaction.overlap_size"].AsInt64() != 1 {
		t.Errorf("overlap_size = %d, want 1", a["gen_ai.compaction.overlap_size"].AsInt64())
	}
	// Only the strategy that fired should be described.
	if _, ok := a["gen_ai.compaction.token_threshold"]; ok {
		t.Error("token_threshold attribute is present on a sliding-window span, want it omitted")
	}
	// The range must be recorded so a trace shows what the summary replaced,
	// and it must be the right range in the right layout. Asserting only that
	// the attributes are non-empty left the layout, the bound each one is
	// sourced from, and the timestamps themselves all unprotected.
	// Epoch seconds as a float, matching the reference implementation. The type
	// is asserted as well as the value, because emitting these as strings is the
	// defect this pins and a string attribute reads back as zero here.
	wantStart := float64(at(1).UnixNano()) / float64(time.Second)
	wantEnd := float64(at(4).UnixNano()) / float64(time.Second)
	if got := a["gen_ai.compaction.start_timestamp"]; got.Type() != attribute.FLOAT64 || got.AsFloat64() != wantStart {
		t.Errorf("start_timestamp = %v (%v), want %v (FLOAT64)", got.Emit(), got.Type(), wantStart)
	}
	if got := a["gen_ai.compaction.end_timestamp"]; got.Type() != attribute.FLOAT64 || got.AsFloat64() != wantEnd {
		t.Errorf("end_timestamp = %v (%v), want %v (FLOAT64)", got.Emit(), got.Type(), wantEnd)
	}
	if got := a["gen_ai.compaction.result_event_id"].AsString(); got == "" {
		t.Error("result_event_id is empty, so a trace cannot be joined to the stored summary")
	}
}

func TestCompactionSpanRecordsFailure(t *testing.T) {
	exp := spanRecorder(t)

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
	}
	cfg := &compaction.Config{CompactionInterval: 2, Summarizer: &fakeSummarizer{err: errors.New("boom")}}

	if _, err := SlidingWindow(context.Background(), cfg, &staticSession{events: events}); err == nil {
		t.Fatal("SlidingWindow() succeeded, want an error")
	}

	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	if spans[0].Status.Code != codes.Error {
		t.Errorf("span status = %v, want %v", spans[0].Status.Code, codes.Error)
	}
	if len(spans[0].Events) == 0 {
		t.Error("span records no exception event, so the failure reason is lost")
	}
	// A failed compaction has no result. Recording one would leave a span that
	// is at once an error and a success, naming an event nothing ever stored.
	a := attrs(spans[0].Attributes)
	for _, key := range []string{
		"gen_ai.compaction.result_event_id",
		"gen_ai.compaction.start_timestamp",
		"gen_ai.compaction.end_timestamp",
	} {
		if _, ok := a[key]; ok {
			t.Errorf("%s is present on a failed compaction span, want it omitted", key)
		}
	}
}

// TestNoSpanWhenNothingToCompact pins that evaluating a trigger and declining is
// silent, so the presence of a span in a trace means compaction really ran.
func TestNoSpanWhenNothingToCompact(t *testing.T) {
	exp := spanRecorder(t)

	events := []*session.Event{textEvent("a", "inv1", 1, "q1")}
	cfg := &compaction.Config{CompactionInterval: 5, Summarizer: &fakeSummarizer{summary: "sum"}}

	got, err := SlidingWindow(context.Background(), cfg, &staticSession{events: events})
	if err != nil || got != nil {
		t.Fatalf("SlidingWindow() = (%v, %v), want (nil, nil)", got, err)
	}
	if n := len(exp.GetSpans()); n != 0 {
		t.Errorf("got %d spans when the interval was not reached, want 0", n)
	}
}

// TestSpanRecordsDecliningSummarizer distinguishes "ran and produced nothing"
// from "ran and failed": the span exists and is successful, but carries no
// result attributes.
func TestSpanRecordsDecliningSummarizer(t *testing.T) {
	exp := spanRecorder(t)

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
	}
	cfg := &compaction.Config{CompactionInterval: 2, Summarizer: &fakeSummarizer{}}

	if _, err := SlidingWindow(context.Background(), cfg, &staticSession{events: events}); err != nil {
		t.Fatalf("SlidingWindow() error = %v", err)
	}

	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	if spans[0].Status.Code == codes.Error {
		t.Errorf("span status = error, want success for a summarizer that merely declined")
	}
	if _, ok := attrs(spans[0].Attributes)["gen_ai.compaction.result_event_id"]; ok {
		t.Error("result_event_id is set although no summary was produced")
	}
}

// bothSummarizer returns a usable compaction event alongside an error, which a
// third-party Summarizer is free to do.
type bothSummarizer struct{}

func (s *bothSummarizer) SummarizeEvents(_ context.Context, events []*session.Event) (*session.Event, error) {
	ev, err := compaction.NewSummaryEvent(events, genai.NewContentFromText("SUM", "model"), nil)
	if err != nil {
		return nil, err
	}
	return ev, errors.New("boom")
}

// TestCompactionSpanOmitsResultWhenSummarizerAlsoErrors pins that a span is
// never both an error and a success.
//
// A Summarizer may return an event and an error together. The caller discards
// the event, so recording its identity would name something no session holds,
// and the span would report a failure while carrying the attributes of a
// success.
func TestCompactionSpanOmitsResultWhenSummarizerAlsoErrors(t *testing.T) {
	exp := spanRecorder(t)

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
	}
	cfg := &compaction.Config{CompactionInterval: 2, Summarizer: &bothSummarizer{}}

	if _, err := SlidingWindow(context.Background(), cfg, &staticSession{events: events}); err == nil {
		t.Fatal("SlidingWindow() succeeded, want an error")
	}

	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	if spans[0].Status.Code != codes.Error {
		t.Errorf("span status = %v, want %v", spans[0].Status.Code, codes.Error)
	}
	a := attrs(spans[0].Attributes)
	for _, key := range []string{
		"gen_ai.compaction.result_event_id",
		"gen_ai.compaction.start_timestamp",
		"gen_ai.compaction.end_timestamp",
	} {
		if v, ok := a[key]; ok {
			t.Errorf("%s = %q on a failed compaction span, want it omitted", key, v.AsString())
		}
	}
}

// panickingSummarizer models third-party code that blows up.
type panickingSummarizer struct{}

func (s *panickingSummarizer) SummarizeEvents(_ context.Context, _ []*session.Event) (*session.Event, error) {
	panic("summarizer exploded")
}

// TestCompactionSpanMarksAPanic pins that a panicking summarizer does not leave
// a span that reads as success.
//
// The OTel SDK records an exception event on the way out but leaves the status
// Unset, and Unset is indistinguishable from a healthy compaction that produced
// nothing. The panic itself still propagates.
func TestCompactionSpanMarksAPanic(t *testing.T) {
	exp := spanRecorder(t)

	events := []*session.Event{
		textEvent("a", "inv1", 1, "q1"), modelTextEvent("b", "inv1", 2, "a1"),
		textEvent("c", "inv2", 3, "q2"), modelTextEvent("d", "inv2", 4, "a2"),
	}
	cfg := &compaction.Config{CompactionInterval: 2, Summarizer: &panickingSummarizer{}}

	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Error("the panic did not propagate; compaction must not swallow it")
			}
		}()
		_, _ = SlidingWindow(context.Background(), cfg, &staticSession{events: events})
	}()

	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	if spans[0].Status.Code != codes.Error {
		t.Errorf("span status = %v, want %v: a panicking summarizer must not look healthy", spans[0].Status.Code, codes.Error)
	}
}
