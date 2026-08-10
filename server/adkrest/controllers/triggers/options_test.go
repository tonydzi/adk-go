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

package triggers

import (
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// TestControllerConstructorTypesAreUnchanged pins the exported *type* of the
// trigger constructors, not merely that a call compiles.
//
// This is the assertion the previous version of this test was missing. A plain
// call expression still compiles after a trailing variadic parameter is added,
// so it cannot catch the one change that actually breaks downstream code:
// anything that referenced the constructor as a value, or stored it in a field
// of that function type, stops compiling. Assigning to an explicit function
// type is what makes the signature part of the contract.
func TestControllerConstructorTypesAreUnchanged(t *testing.T) {
	t.Parallel()

	if got := pubSubCtor(nil, nil, nil, nil, runner.PluginConfig{}, TriggerConfig{MaxConcurrentRuns: 1}); got == nil {
		t.Error("NewPubSubController() returned nil")
	}
	if got := eventarcCtor(nil, nil, nil, nil, runner.PluginConfig{}, TriggerConfig{MaxConcurrentRuns: 1}); got == nil {
		t.Error("NewEventarcController() returned nil")
	}
}

// The declared types are the assertion: assigning each constructor to an
// explicit function type fails to compile if its signature changes, including
// by gaining a trailing variadic parameter, which an ordinary call expression
// would still accept.
var (
	pubSubCtor   NewPubSubControllerFunc   = NewPubSubController
	eventarcCtor NewEventarcControllerFunc = NewEventarcController
)

// NewPubSubControllerFunc is the released signature of [NewPubSubController].
type NewPubSubControllerFunc = func(session.Service, agent.Loader, memory.Service, artifact.Service, runner.PluginConfig, TriggerConfig) *PubSubController

// NewEventarcControllerFunc is the released signature of [NewEventarcController].
type NewEventarcControllerFunc = func(session.Service, agent.Loader, memory.Service, artifact.Service, runner.PluginConfig, TriggerConfig) *EventarcController

func TestWithEventsCompactionConfig(t *testing.T) {
	t.Parallel()

	cfg := &compaction.Config{CompactionInterval: 3, OverlapSize: 1}
	tc := TriggerConfig{MaxConcurrentRuns: 1}

	tests := []struct {
		name   string
		runner *RetriableRunner
	}{
		{
			name:   "pubsub",
			runner: NewPubSubControllerWithOptions(nil, nil, nil, nil, runner.PluginConfig{}, tc, WithEventsCompactionConfig(cfg)).runner,
		},
		{
			name:   "eventarc",
			runner: NewEventarcControllerWithOptions(nil, nil, nil, nil, runner.PluginConfig{}, tc, WithEventsCompactionConfig(cfg)).runner,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if tt.runner.eventsCompactionConfig != cfg {
				t.Errorf("eventsCompactionConfig = %v, want the config passed to the option", tt.runner.eventsCompactionConfig)
			}
		})
	}
}

func TestWithEventsCompactionConfigDefaultsToNil(t *testing.T) {
	t.Parallel()

	c := NewPubSubController(nil, nil, nil, nil, runner.PluginConfig{}, TriggerConfig{MaxConcurrentRuns: 1})
	if c.runner.eventsCompactionConfig != nil {
		t.Errorf("eventsCompactionConfig = %v, want nil when the option is not supplied", c.runner.eventsCompactionConfig)
	}
}

// TestControllerOptionsToleratesNil checks that a nil option is skipped rather
// than dereferenced.
//
// Options are commonly assembled by a helper that returns nil when it has
// nothing to apply, and a variadic parameter makes passing one easy. Panicking
// during construction is a poor way to report that.
func TestControllerOptionsToleratesNil(t *testing.T) {
	t.Parallel()

	if got := NewPubSubControllerWithOptions(nil, nil, nil, nil, runner.PluginConfig{}, TriggerConfig{MaxConcurrentRuns: 1}, nil); got == nil {
		t.Error("NewPubSubController() with a nil option returned nil")
	}
	if got := NewEventarcControllerWithOptions(nil, nil, nil, nil, runner.PluginConfig{}, TriggerConfig{MaxConcurrentRuns: 1}, nil); got == nil {
		t.Error("NewEventarcController() with a nil option returned nil")
	}
	// A nil option alongside a real one must not stop the real one applying.
	cfg := &compaction.Config{CompactionInterval: 2}
	c := NewPubSubControllerWithOptions(nil, nil, nil, nil, runner.PluginConfig{}, TriggerConfig{MaxConcurrentRuns: 1}, nil, WithEventsCompactionConfig(cfg))
	if c.runner.eventsCompactionConfig != cfg {
		t.Error("a nil option prevented a later option from applying")
	}
}
