"""Frozen scope; no model, environment, game or additional sample execution."""
BASE = '596c2767342e8311b1272a417c2a83782c29423a'
PRIOR_COMMIT = 'de61ad753204a9234fb447cdd0fe25b1bf99e378'
ENGINE_COMMIT = 'f994f2050d58d84f649329c1e35ce09ada7007c5'
ENGINE_SHA256 = 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
PRIOR_COMMITMENT = '566ae9ccd0d3135c7a7342d98086d8c628d631c9d6eeae1605159cae5140205a'
ROLES = ('F01', 'F02', 'F03', 'F04')
GIB, MIB = 1024**3, 1024**2
LIMITS = dict(initial_available_bytes=4*GIB, running_available_bytes=4*GIB,
    max_rss_bytes=2*GIB, wall_seconds=900, max_private_bytes=16*MIB,
    closure_reserve_bytes=MIB, max_output_file_bytes=8*MIB, max_line_bytes=8*MIB,
    max_compressed_stream_bytes=128*MIB, max_decoded_stream_bytes=512*MIB,
    max_decoded_total_bytes=2*GIB, cases=4, arms=8, suffix_transitions=384,
    stream_passes_per_case=1, max_attempts=1)


def specification(analysis):
    return dict(scope='four_consumed_48_step_pairs_read_only_trajectory_analysis',
        analysis=analysis, cases=list(ROLES), horizon=48,
        inputs='45 sealed originals: 41 small metadata files and 4 event streams',
        preparation='41 small originals authenticated; 4 gzip bodies unopened, sizes checked',
        run='one gzip pass per case: compressed SHA plus CRC/EOF plus analysis',
        source_history_reads=0, source_game_inventory_reads=0, model_forwards=0,
        official_transitions=0, environment_constructions=0, complete_games=0,
        new_seeds=False, training=False, gpu=False, independent_evaluation=False,
        statistical_significance=False, promotion=False, retries=False,
        later_market_requests_are_not_trades=True,
        causal_price_volume_scheduling_percentages=False,
        peer_result_review='code and compact derived ledgers only; no second gzip pass',
        local_registration_not_third_party_timestamp=True)
