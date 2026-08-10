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

package llminternal

import (
	"fmt"
	"iter"
	"log"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agent/compactionctx"
	"google.golang.org/adk/v2/internal/compactioninternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// CompactionRequestProcessor runs token-threshold tail-retention compaction
// before the conversation history is assembled for a model call.
//
// It must sit before [ContentsRequestProcessor] in the chain: the summary it
// appends only shrinks this request if contents are built afterwards.
//
// Unlike the runner's post-invocation sliding-window pass, this runs mid-turn,
// so it can react to a single long-running invocation that inflates the prompt
// rather than waiting for the turn to finish. It leaves the request itself
// untouched and emits no events.
func CompactionRequestProcessor(ctx agent.InvocationContext, _ *model.LLMRequest, _ *Flow) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		rt := compactionctx.FromContext(ctx)
		if !rt.Enabled() {
			return
		}
		if ctx.Session() == nil {
			return
		}

		// Compact against the session underneath any wrapper an agent installed
		// over it. A wrapper carries a synthetic first-turn seed that no store
		// holds, and every session service type-asserts on its own concrete
		// type, so appending a summary to one fails outright.
		//
		// Unwrapping rather than re-reading keeps object identity with the
		// session the wrapper reads through, so the summary appended below
		// reaches the prompt this processor runs ahead of. A freshly read
		// session would be a different object and the summary would miss it.
		sess := compactioninternal.UnwrapSession(ctx.Session())

		// Compaction is an optimisation, so a cancelled or expired turn should
		// not spend a model call on it.
		if ctx.Err() != nil {
			return
		}
		summary, err := compactioninternal.TailRetention(ctx, rt.Config, sess, promptTokenEstimator(ctx))
		if err != nil {
			// Surfaced rather than swallowed, unlike the post-invocation pass.
			// Compaction fires here precisely because the prompt is already
			// near the context limit, so continuing would most likely fail the
			// model call anyway, with a far less informative error.
			yield(nil, compactionFailure("token-threshold", err))
			return
		}
		if summary == nil {
			return
		}

		// Summarizing takes a model call, which is long enough for another
		// invocation on this session to append inside the range just chosen.
		// Read the stored session and abandon the summary if anything landed
		// inside it. Skipping costs one wasted call, where recording it would
		// silently drop those turns from every later prompt.
		//
		// The read is only a comparison. The append below still goes to sess,
		// for the identity reason above.
		if ctx.Err() != nil {
			return
		}
		latest, err := compactioninternal.ReloadSession(ctx, rt.SessionService, sess)
		if err != nil {
			yield(nil, compactionFailure("token-threshold", err))
			return
		}
		if compactioninternal.RangeRaced(latest, sess, summary) {
			log.Printf("adk: discarding a tail-retention summary because the session changed inside its range while summarizing")
			return
		}

		if err := rt.SessionService.AppendEvent(ctx, sess, summary); err != nil {
			yield(nil, compactionFailure("failed to append the summary event", err))
		}
	}
}

// compactionFailure marks err as a compaction failure at the named stage.
//
// The cause is rendered with %v rather than wrapped with %w, deliberately. This
// error is yielded into the flow's error channel, which reaches the workflow
// scheduler, and the scheduler tests for a context.Canceled chain before
// anything else and drops the error when it finds one. A summarizer that failed
// because its own context was cancelled would therefore end the turn with no
// answer, no events and no error at all: the most confusing outcome available.
//
// Cutting the chain keeps the cause in the message and keeps the error matchable
// as [compaction.ErrCompaction], at the cost of errors.Is against the cause. For
// a bookkeeping failure that is the right way round.
func compactionFailure(stage string, err error) error {
	return fmt.Errorf("%w: %s: %v", compaction.ErrCompaction, stage, err)
}

// promptTokenEstimator returns a [compactioninternal.TokenCounter] that approximates
// the prompt size for ctx's agent.
//
// It is only consulted before any model response has reported a real token
// count. Building the contents the same way the request will is what makes the
// estimate meaningful: it sees branch and isolation-scope filtering, and any
// compaction already applied.
func promptTokenEstimator(ctx agent.InvocationContext) compactioninternal.TokenCounter {
	return func(events []*session.Event) int {
		llmAgent := asLLMAgent(ctx.Agent())
		if llmAgent == nil {
			return 0
		}
		state := llmAgent.internal()
		contents, err := buildContentsDefault(
			ctx.Agent().Name(),
			ctx.Branch(),
			ctx.IsolationScope(),
			events,
			state.Mode == ModeSingleTurn,
			ctx.UserContent(),
		)
		if err != nil {
			// An unbuildable history is the contents processor's problem to
			// report a moment from now; here it just means "no estimate".
			return 0
		}

		return compactioninternal.EstimateTokensFromContents(contents)
	}
}
