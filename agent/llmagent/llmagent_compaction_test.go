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

package llmagent_test

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/httprr"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// TestCompactionE2E drives a real model through enough turns to trigger a
// sliding-window compaction, then checks that the next prompt is both smaller
// and still accepted.
//
// What a passing run does and does not establish. Replay keys on exact request
// bytes, so this proves the recorded model accepted the compacted prompt at the
// time it was recorded. It does not re-validate anything against a live API: a
// structural defect introduced later surfaces as a cassette miss, not as an API
// rejection, and the miss aborts the run before any assertion below is reached.
// The structural properties are therefore asserted here directly and offline,
// over the prompts the agent actually sent.
//
// One property is genuinely not covered. The recorded conversation issues no
// tool call after the compaction point, so the final prompt carries no function
// traffic at all and the call-pairing and call-recovery paths are inert against
// this cassette. Those are covered offline in internal/compactioninternal.
// Covering them here would need a re-record whose fourth turn calls a tool after
// the summary exists.
//
// Deliberately not asserted: any particular wording for the summary. It is model
// output, and pinning it would fail on any model or prompt revision without
// indicating a real problem. What is asserted is that whatever the summarizer
// produced is what reaches the next prompt.
//
// Recording: this test needs a cassette. With credentials available, run
//
//	GOOGLE_API_KEY=... go test ./agent/llmagent/ \
//	    -run '^TestCompactionE2E$' -httprecord='TestCompactionE2E\.httprr$' -count=1 -v
//
// Note the two regexes differ on purpose. -run matches test names, so it is
// anchored. -httprecord matches the cassette FILE PATH, so anchoring it the same
// way would never match "testdata/TestCompactionE2E.httprr" and this test would
// silently skip instead of recording.
//
// and commit the resulting testdata/TestCompactionE2E.httprr. Until then the
// test skips.
//
// This test deliberately has no //go:generate directive of its own. The
// package-level one already carries -httprecord=Test, which matches every
// cassette here, so adding a third changed nothing except the number of ways to
// re-record all of them by accident. For the same reason, do not record with
// "go generate ./agent/llmagent/...". Note also that a failed
// recording still leaves a cassette behind, and it can look plausibly sized
// because the failing exchange is recorded too. Delete it, or the next run finds
// a file, declines to skip, and replays the recorded failure.
//
// The cassette is sensitive to anything that changes prompt bytes, including the
// summarizer prompt template, the transcript line format, tool-argument
// rendering and truncation behaviour. Any of those changes requires re-recording.
func TestCompactionE2E(t *testing.T) {
	// Matches llmagent_delegation_test.go, which is the most recently recorded
	// suite in this package. Change it only alongside a re-record, since the
	// model name is part of the request URL the cassette keys on.
	const compactionModelName = "gemini-3.5-flash"

	// The cassette is committed, so its absence is a lost or renamed file rather
	// than an unrecorded checkout. Skipping would turn that into a silent pass,
	// which is how a test quietly stops running for months. Every other cassette
	// test in this package fails instead, and so does this one. Recording mode
	// goes ahead regardless, since that is the run that creates the file.
	trace := filepath.Join("testdata", t.Name()+".httprr")
	if recording, _ := httprr.Recording(trace); !recording {
		if _, err := os.Stat(trace); err != nil {
			t.Fatalf("no cassette at %s: %v. It is committed, so this means it was lost or renamed. "+
				"Re-record with: GOOGLE_API_KEY=... go test ./agent/llmagent/ "+
				"-run '^TestCompactionE2E$' -httprecord='TestCompactionE2E\\.httprr$' -count=1 -v", trace, err)
		}
	}

	// Captured before each model call, so the assertions can look at the exact
	// history the agent sent rather than inferring it from the session.
	var (
		mu      sync.Mutex
		prompts [][]*genai.Content
		capture = func(_ agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
			mu.Lock()
			defer mu.Unlock()
			prompts = append(prompts, req.Contents)
			return nil, nil
		}
	)

	// A tool gives the transcript function calls and responses to render, which
	// is the part of the summarizer prompt most likely to break.
	type cityArgs struct {
		City string `json:"city" jsonschema:"the city to look up"`
	}
	type weatherResult struct {
		Weather string `json:"weather"`
	}
	weather, err := functiontool.New[cityArgs, weatherResult](
		functiontool.Config{Name: "get_weather", Description: "Returns the weather in a city."},
		func(_ agent.Context, args cityArgs) (weatherResult, error) {
			return weatherResult{Weather: "sunny in " + args.City}, nil
		},
	)
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:                     "compaction_agent",
		Description:              "agent used to exercise context compaction",
		Model:                    newGeminiModel(t, compactionModelName, nil),
		Instruction:              "You are a concise assistant. Answer in one short sentence.",
		Tools:                    []tool.Tool{weather},
		BeforeModelCallbacks:     []llmagent.BeforeModelCallback{capture},
		DisallowTransferToParent: true,
		DisallowTransferToPeers:  true,
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}

	// Interval 2 keeps the recording short: three turns produce exactly one
	// compaction, which is all this test needs.
	//
	// No OverlapSize. It would be inert here and claiming otherwise was wrong:
	// overlap only reaches back past an earlier compaction, and the first
	// compaction of a session starts at the first invocation whatever the
	// overlap is. Exercising the seam needs a second window, so it belongs in
	// the offline tests where windows are cheap.
	r := testutil.NewTestAgentRunnerWithCompaction(t, a, &compaction.Config{
		CompactionInterval: 2,
	})

	const sessionID = "compaction_session"
	turns := []string{
		"What is the weather in Zurich?",
		"My favourite colour is teal, remember that.",
		"What was my favourite colour again?",
	}
	var lastAnswer []string
	for i, turn := range turns {
		answer, err := testutil.CollectTextParts(r.Run(t, sessionID, turn))
		if err != nil {
			t.Fatalf("turn %d (%q) failed: %v", i+1, turn, err)
		}
		lastAnswer = answer
	}

	// A compaction must have landed. Without this the rest proves nothing.
	events := sessionEventsFor(t, r, sessionID)
	summaries := make([]*session.Event, 0, 1)
	for _, ev := range events {
		if compaction.IsCompactionEvent(ev) {
			summaries = append(summaries, ev)
		}
	}
	if len(summaries) == 0 {
		t.Fatalf("no compaction event after %d turns, so this test exercised nothing", len(turns))
	}

	summaryText := textOf(summaries[len(summaries)-1].Actions.Compaction.CompactedContent)
	if strings.TrimSpace(summaryText) == "" {
		t.Error("the stored summary is empty")
	}

	// The final prompt is the interesting one: it is the first assembled after a
	// summary existed. It must carry the summary instead of the turns it covers,
	// and the model must have accepted it, which the absence of an error above
	// already establishes.
	mu.Lock()
	defer mu.Unlock()
	if len(prompts) < len(turns) {
		t.Fatalf("captured %d prompts, want at least %d", len(prompts), len(turns))
	}
	final := promptTextOf(prompts[len(prompts)-1])

	if !strings.Contains(final, strings.TrimSpace(summaryText)) {
		t.Errorf("final prompt does not contain the stored summary.\nsummary:\n%s\n\nprompt:\n%s", summaryText, final)
	}
	if strings.Contains(final, turns[0]) {
		t.Errorf("final prompt still contains the compacted first turn %q:\n%s", turns[0], final)
	}
	if !strings.Contains(final, turns[len(turns)-1]) {
		t.Errorf("final prompt is missing the current turn %q:\n%s", turns[len(turns)-1], final)
	}
	// Turn 2 is inside the compacted range as well, so it must be gone for the
	// same reason turn 1 is. Asserting only turn 1 left half the range unchecked.
	if strings.Contains(final, turns[1]) {
		t.Errorf("final prompt still contains the compacted second turn %q:\n%s", turns[1], final)
	}

	// The point of compacting rather than truncating: the fact survives into the
	// summary and the model can still answer from it. Without this the test
	// proved the prompt shrank, not that it kept working.
	answer := strings.ToLower(strings.Join(lastAnswer, " "))
	if !strings.Contains(answer, "teal") {
		t.Errorf("the model could not recall the colour from the summary alone; answer was %q", answer)
	}

	// Structural checks, asserted here rather than inferred from the fact that a
	// recorded model once accepted the bytes. Every prompt is checked, not only
	// the final one.
	for i, p := range prompts {
		assertNoOrphanFunctionResponses(t, i, p)
	}
}

// assertNoOrphanFunctionResponses checks that every function response in a
// prompt is preceded by the call it answers.
//
// This is the property compaction is most likely to break: the summary replaces
// a span of history, and a response whose call fell inside that span while the
// response itself did not would reach the model unpaired, which real backends
// reject.
func assertNoOrphanFunctionResponses(t *testing.T, promptIdx int, contents []*genai.Content) {
	t.Helper()

	seenCalls := make(map[string]bool)
	responses := 0
	for _, c := range contents {
		if c == nil {
			continue
		}
		for _, part := range c.Parts {
			if part == nil {
				continue
			}
			if fc := part.FunctionCall; fc != nil {
				seenCalls[callKey(fc.ID, fc.Name)] = true
			}
			if fr := part.FunctionResponse; fr != nil {
				responses++
				if !seenCalls[callKey(fr.ID, fr.Name)] {
					t.Errorf("prompt %d carries a function response %q with no preceding call", promptIdx, fr.Name)
				}
			}
		}
	}
	if responses == 0 {
		// Not a failure. It is the documented gap: this recording has no tool
		// traffic after the compaction point, so for the final prompt there is
		// nothing here to check.
		t.Logf("prompt %d carries no function responses, so pairing was not exercised in it", promptIdx)
	}
}

// callKey identifies a call by ID when the model supplies one, and by name
// otherwise, which is what happens with models that omit call IDs.
func callKey(id, name string) string {
	if id != "" {
		return "id:" + id
	}
	return "name:" + name
}

func sessionEventsFor(t *testing.T, r *testutil.TestAgentRunner, sessionID string) []*session.Event {
	t.Helper()
	resp, err := r.SessionService().Get(context.Background(), &session.GetRequest{
		AppName: "test_app", UserID: "test_user", SessionID: sessionID,
	})
	if err != nil {
		t.Fatalf("session Get() error = %v", err)
	}
	var events []*session.Event
	for ev := range resp.Session.Events().All() {
		events = append(events, ev)
	}
	return events
}

func textOf(c *genai.Content) string {
	if c == nil {
		return ""
	}
	var b strings.Builder
	for _, p := range c.Parts {
		if p != nil && p.Text != "" {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

func promptTextOf(contents []*genai.Content) string {
	var b strings.Builder
	for _, c := range contents {
		if c == nil {
			continue
		}
		for _, p := range c.Parts {
			switch {
			case p == nil:
			case p.Text != "":
				b.WriteString("[" + c.Role + "] " + p.Text + "\n")
			case p.FunctionCall != nil:
				b.WriteString("[" + c.Role + "] CALL " + p.FunctionCall.Name + "\n")
			case p.FunctionResponse != nil:
				b.WriteString("[" + c.Role + "] RESPONSE " + p.FunctionResponse.Name + "\n")
			}
		}
	}
	return b.String()
}
