abstract type WeightMatrix end

mutable struct BlockBoostrapWeightMatrix <: WeightMatrix
    seed::Int64
    obs::Array{Float64,1}
    block_size::Int64
    n_bootstap::Int64
    weights::Array{Float64, 2}
end
function BlockBoostrapWeightMatrix(seed::Int64, obs::Array{Float64, 2},
    block_size::Int64, n_bootstrap::Int64)
    # Step 1: Apply a Moving Block Bootstrap to the Measured Series
    Random.seed!(seed)
    n = size(obs, 1)
    block_ind = 1:n-block_size+1
    b_samples = Array{Float64,2}(undef, n, n_bootstap)
    for i in 1:n_bootstap
        b_samples[:,i] = obs[block_bootstrap_index(block_ind, n, block_size)]
    end

    # Step 2: Calculate Distributions for Each Moment and Test Statistic
    dist = get_summary_stats(b_samples, obs)

    # Step 3: Calculate Inverse Weight Matrix
    W = inv(cov(dist))
    return BlockBoostrapWeightMatrix(seed, obs, block_size, n_bootstrap, W)
end


function block_bootstrap_index(block_ind, n, b)
    rand_blocks = sample(block_ind, floor(Int,n/b))
    sample_ind = transpose(repeat(rand_blocks, 1, b))
    sample_ind = sample_ind[:]
    addition_vec = repeat(0:b-1,floor(Int,n/b))
    sample_ind = sample_ind + addition_vec
    return sample_ind
end

function select_moments(obs)
    sample_mean = mean(obs)
    sample_std = std(obs)
    sample_kurt = normal_kurtosis(obs)
    sample_ks_stat = 0.0
    sample_hurst = generalized_hurts_exp(obs)

    return [sample_mean, sample_std, sample_kurt, sample_ks_stat, sample_hurst]
end
