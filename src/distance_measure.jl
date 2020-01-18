function weighted_moment_distance(wm::BlockBoostrapWeightMatrix, sims)
    observed_moments = select_moments(wm.obs)

    n_replications = size(sims, 2)
    G = zeros(n_replications, 5)

    sim_moments_matrix = get_summary_stats(sims, wm.obs)
    G = sim_moments_matrix - repeat(observed_moments', n_replications, 1)
    G = mean(G, dims = 1)
    return (G * wm.weights * G')[]
end
