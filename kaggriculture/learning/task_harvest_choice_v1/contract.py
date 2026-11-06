"""One registered four-scenario official interface validation, not an evaluation."""
BASE = 'f4d52334015f449480bda67726a69543d124ead5'
PRIOR_COMMIT = '5df6a27af653ed6396f34118840427facffbcb2f'
ENGINE_COMMIT = 'f994f2050d58d84f649329c1e35ce09ada7007c5'
ENGINE_SHA = 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
CORE_SHA = '0922c4599a1b6e0d8c3dadf06ae5297f98138d859d686f00daee6f36e6d45d0e'
SPEC_SHA = 'a82c89c1a2315b93f39775d8e025471a01b738647c9772658368ee6b1b6f4867'
UTILS_SHA = '537b627b11784d424147ef57ebb0369b039bf83c9f891e81f10486b1f552334b'
TEACHER_SHA = '0844f0152e742aba684108b5914964fb85082b3c0881d0da4d41649907a9b6ff'
GIB, MIB = 1024**3, 1024**2
LIMITS = dict(initial_available_bytes=4*GIB, running_available_bytes=4*GIB,
    max_rss_bytes=2*GIB, wall_seconds=900, max_private_bytes=16*MIB,
    closure_reserve_bytes=MIB, max_output_file_bytes=4*MIB,
    max_environment_initializations=8, max_action_transitions=8,
    max_attempts=1, complete_games=0, model_forwards=0)
PROBE_CAPS = dict(runs=1, initializations=8, action_steps=8,
    agent_constructions=6, agent_calls=6, selector_calls=6, emissions=8)


def specification(fixtures):
    return dict(scope='direct_melon_water_harvest_choice_expression_only',
        fixtures=fixtures, candidate_scope='existing eligible WATER plus direct legal MELON HARVEST',
        disabled_or_original_id_complete_request_parity=True,
        original_vectors_order_and_shared_inputs_preserved=True,
        virtual_options_do_not_increase_real_task_count=True,
        selected_virtual_option_replaces_original_task_before_consumption=True,
        original_teacher_label_is_reference_not_selector_input=True,
        no_profitability_or_storage_capacity_mask=True,
        explicit_test_selectors_not_trained_models=True,
        synthetic_scenarios_not_natural_trajectories=True,
        counter_placeholders_are_not_replay=True,
        max_environment_initializations=8, max_action_transitions=8,
        probe_caps=PROBE_CAPS, direct_boundary_requests=2, evidence_write_failures=0,
        old_replay_trace_gzip_reads=0, game_seed_inventory_reads=0,
        model_weight_reads=0, model_forwards=0, complete_games=0,
        training=False, gpu=False, independent_evaluation=False,
        future_return_evaluated=False, promotion=False, retries=False,
        local_registration_not_third_party_timestamp=True)
