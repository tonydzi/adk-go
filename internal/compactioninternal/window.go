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
	"fmt"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
)

// longestSelfContainedPrefix returns the longest prefix of events that is safe
// to summarize.
//
// A single left-to-right pass tracks "open" obligations keyed by call ID: a
// function call, or a tool-confirmation request, opens one; a function response
// with the same ID closes it. Responses are applied before calls within one
// event, so a response only ever closes an obligation opened by an earlier
// event. Summarizing is safe exactly at the points where nothing is open, so
// the prefix ending at the last such point is returned.
//
// The result is empty when the window never reaches a balanced point, which
// tells the caller to skip this compaction rather than strand a half-finished
// tool interaction. Without this, a summary could swallow a function call while
// leaving its response behind, which downstream prompt assembly rejects.
//
// The prefix is additionally pulled back off a timestamp tie. Compaction
// coverage is an inclusive timestamp range, so if the first excluded event
// shares a timestamp with the last included one, it would fall inside the
// summarized range without having been summarized, and disappear from the
// prompt. Cutting before the whole tied group keeps "summarized" and "covered"
// the same set.
func longestSelfContainedPrefix(events []*session.Event) []*session.Event {
	openIDs := make(map[string]struct{})
	safeLength := 0
	for i, ev := range events {
		for _, resp := range utils.FunctionResponses(utils.Content(ev)) {
			delete(openIDs, resp.ID)
		}
		for _, call := range utils.FunctionCalls(utils.Content(ev)) {
			openIDs[callObligationKey(call, i)] = struct{}{}
		}
		for id := range ev.Actions.RequestedToolConfirmations {
			openIDs[id] = struct{}{}
		}
		// TODO: track outstanding authentication requests here too once
		// adk-go models them on EventActions.
		if len(openIDs) == 0 {
			safeLength = i + 1
		}
	}
	return events[:trimToTimestampBoundary(events, safeLength)]
}

// callObligationKey returns the key a function call is tracked under while
// waiting for its response.
//
// An ID-less call gets a synthetic key that no response can match, so it stays
// open forever and the prefix is cut before it. FunctionCall.ID is optional and
// some providers omit it, and keying such a call on "" would let a single
// unrelated ID-less response close it, or -- worse, and what the earlier
// implementation did -- skip it entirely so the trim that protects every other
// call silently never fires. Refusing to summarize is the safe direction.
func callObligationKey(call *genai.FunctionCall, eventIndex int) string {
	if call.ID != "" {
		return call.ID
	}
	return fmt.Sprintf("\x00no-id\x00%d\x00%s", eventIndex, call.Name)
}

// trimToTimestampBoundary pulls length back so the cut does not fall inside a
// group of events sharing a timestamp.
func trimToTimestampBoundary(events []*session.Event, length int) int {
	if length <= 0 || length >= len(events) {
		return length
	}
	boundary := events[length].Timestamp
	for length > 0 && !events[length-1].Timestamp.Before(boundary) {
		length--
	}
	return length
}

// LatestCompactionEvent returns the newest compaction event in events that no
// other compaction subsumes, or nil when events holds no compaction at all.
//
// A compaction is subsumed when another compaction fully contains its range: a
// strictly wider range, or an identical range appearing later in the stream.
//
// Ties are broken by stream position rather than by greatest end timestamp,
// because the summary written later saw more history and supersedes the earlier
// one even when both cover the same range.
func LatestCompactionEvent(events []*session.Event) *session.Event {
	var latest *session.Event
	for i, ev := range events {
		if !hasCompaction(ev) {
			continue
		}
		if isCompactionSubsumed(i, ev.Actions.Compaction, events) {
			continue
		}
		latest = ev
	}
	return latest
}

// isCompactionSubsumed reports whether the compaction at index i is fully
// contained by another compaction in events. Identical ranges are broken by
// stream position: the earlier event is subsumed by the later one.
func isCompactionSubsumed(i int, rng *session.EventCompaction, events []*session.Event) bool {
	for j, other := range events {
		if j == i || !hasCompaction(other) {
			continue
		}
		o := other.Actions.Compaction
		if o.StartTimestamp.After(rng.StartTimestamp) || o.EndTimestamp.Before(rng.EndTimestamp) {
			continue
		}
		if o.StartTimestamp.Before(rng.StartTimestamp) || o.EndTimestamp.After(rng.EndTimestamp) || j > i {
			return true
		}
	}
	return false
}

// selectSlidingWindow returns the events a sliding-window compaction should
// summarize, or nil when there is nothing to compact yet.
//
// The window is a *contiguous slice* of the event list, from the first event of
// the oldest invocation being compacted through the last event of the newest.
// Contiguity is the point: compaction coverage is recorded as an inclusive
// timestamp range, and the prompt builder drops every event inside that range.
// Building the window by filtering instead would let an event be skipped by the
// filter yet still fall inside the range, so it would be dropped from the
// prompt without ever having been summarized. Slicing makes that unexpressible.
//
// Which invocations to cover is decided first, then the slice is taken. An
// invocation counts as new when it has any event after the most recent
// compaction boundary. Once interval new invocations exist, the window reaches
// back overlap further invocations so consecutive summaries share context.
//
// nil comes back when fewer than interval new invocations exist, or when the
// slice has no self-contained prefix left after trimming.
func selectSlidingWindow(events []*session.Event, interval, overlap int) []*session.Event {
	if interval <= 0 {
		return nil
	}

	// The boundary of the newest compaction already recorded. Everything at or
	// before it has been summarized once already.
	var lastCompactEnd time.Time
	for _, ev := range events {
		if hasCompaction(ev) {
			if end := ev.Actions.Compaction.EndTimestamp; end.After(lastCompactEnd) {
				lastCompactEnd = end
			}
		}
	}

	// Invocations in first-seen order, and whether each has any event past the
	// boundary. hasCompaction rather than IsCompactionEvent: an event declaring
	// a compaction is bookkeeping even when its content is unusable, and must
	// never be counted as a conversational invocation.
	var order []string
	isNew := make(map[string]bool)
	for _, ev := range events {
		if hasCompaction(ev) || ev.InvocationID == "" {
			continue
		}
		if _, ok := isNew[ev.InvocationID]; !ok {
			order = append(order, ev.InvocationID)
			isNew[ev.InvocationID] = false
		}
		if ev.Timestamp.After(lastCompactEnd) {
			isNew[ev.InvocationID] = true
		}
	}

	firstNew := -1
	newCount := 0
	for i, id := range order {
		if isNew[id] {
			if firstNew < 0 {
				firstNew = i
			}
			newCount++
		}
	}
	if firstNew < 0 || newCount < interval {
		return nil
	}

	// Cover at most interval new invocations, rather than running to the end of
	// the session.
	//
	// Uncapped, the window is O(session) instead of O(interval): the first
	// compaction after enabling the feature on an existing deployment would
	// hand a whole live conversation to one model call, which can exceed the
	// summarizer's own context limit. It also compounds, because a summarizer
	// error records nothing, so the next turn recomputes from the same start
	// over a strictly larger window and is more likely to fail again. Capping
	// makes a retry the same size as the attempt that failed, and drains any
	// backlog one bounded window per turn.
	startID := order[max(0, firstNew-overlap)]
	endID := order[min(len(order)-1, firstNew+interval-1)]

	// Slice from the first event of startID through the last of endID. Events
	// in between are included whatever they are, including ones with no
	// invocation ID, which is exactly the contiguity the range model needs.
	first, last := -1, -1
	for i, ev := range events {
		if hasCompaction(ev) {
			continue
		}
		if first < 0 && ev.InvocationID == startID {
			first = i
		}
		if ev.InvocationID == endID {
			last = i
		}
	}
	if first < 0 || last < first {
		return nil
	}

	window := make([]*session.Event, 0, last-first+1)
	for _, ev := range events[first : last+1] {
		// Prior summaries are bookkeeping, not conversation, and are the only
		// thing dropped from the slice. They are never re-summarized, so a
		// sliding-window compaction is a constant-factor reduction rather than
		// a bound; the tail-retention strategy is what bounds prompt growth.
		if hasCompaction(ev) {
			continue
		}
		window = append(window, ev)
	}

	// A summary inherits the branch and isolation scope of what it covers, so
	// the window has to be homogeneous in both. A contiguous slice of a
	// multi-agent session routinely spans branches, and summarizing across one
	// would fold a sub-agent's content into a summary visible to the parent,
	// defeating the filters that keep those separate.
	window = trimToOneScope(window)

	if trimmed := longestSelfContainedPrefix(window); len(trimmed) > 0 {
		return trimmed
	}
	return skipBlockedHead(window)
}

// trimToOneScope cuts the window at the first event whose branch or isolation
// scope differs from the first event's.
func trimToOneScope(window []*session.Event) []*session.Event {
	if len(window) == 0 {
		return window
	}
	branch, scope := window[0].Branch, window[0].IsolationScope
	for i, ev := range window {
		if ev.Branch != branch || ev.IsolationScope != scope {
			return window[:i]
		}
	}
	return window
}

// skipBlockedHead handles a window whose very first events hold a function call
// that never got a response, which leaves no self-contained prefix at all.
//
// A tool awaiting human approval, or one whose backend died, blocks the head of
// the window permanently. Because the window is anchored to the last compaction
// boundary, that call stays at the head on every later attempt, so compaction
// would stop for the rest of the session and, since "no prefix" and "not enough
// invocations yet" both come back as nil, do so silently. Long tool-using
// sessions are exactly the ones compaction exists for.
//
// So instead of giving up, step past the blocked head and summarize the longest
// self-contained run that follows. The blocked call and everything before it
// stay raw and visible, which is what a pending call needs anyway. The summary
// is a contiguous later range, so the coverage invariant still holds.
//
// nil still comes back when nothing after the blockage is self-contained
// either.
func skipBlockedHead(window []*session.Event) []*session.Event {
	for start := 1; start < len(window); start++ {
		// Only resume just after an event that opened an obligation, so the
		// scan is over blockage points rather than every offset.
		prev := window[start-1]
		if len(utils.FunctionCalls(utils.Content(prev))) == 0 && len(prev.Actions.RequestedToolConfirmations) == 0 {
			continue
		}
		if tail := longestSelfContainedPrefix(window[start:]); len(tail) > 0 {
			return tail
		}
	}
	return nil
}
