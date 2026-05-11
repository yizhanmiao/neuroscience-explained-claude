### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# This Pluto notebook file is @version_header_placeholder

# ╔═╡ 00000000-0000-0000-0000-000000000001
begin
    using PlutoUI
    using Plots
    using Random
    using Statistics
    using StatsBase
end

# ╔═╡ 11111111-0000-0000-0000-000000000001
md"""
# 🧠 Cross-Correlation & Shift Predictor
### An Interactive Tutorial in Computational Neuroscience

This notebook walks you through two foundational tools for studying **neural population activity**:

1. **Cross-Correlation** — measuring co-firing relationships between neuron pairs
2. **Shift Predictor** — separating true network interactions from stimulus-driven correlations

Use the sliders throughout to explore how parameters affect the results.

---
"""

# ╔═╡ 11111111-0000-0000-0000-000000000002
md"""
## 🔬 Part 1 — What Is a Spike Train?

A **spike train** is the sequence of times at which a neuron fires an action potential.
We represent it as a binary vector: `1` at each firing time, `0` otherwise.

Below we simulate a simple **Poisson spike train** — the most common baseline model,
where spikes occur independently at a constant average rate `λ` (in Hz).
"""

# ╔═╡ 11111111-0000-0000-0000-000000000003
md"""
**Firing rate (λ):** $(@bind lambda_demo Slider(1:1:60; default=20, show_value=true)) Hz

**Duration:** $(@bind duration_demo Slider(100:100:2000; default=500, show_value=true)) ms

**Random seed:** $(@bind seed_demo Slider(1:50; default=1, show_value=true))
"""

# ╔═╡ 11111111-0000-0000-0000-000000000004
begin
    """
        poisson_spike_train(
            rate_hz::Real,
            duration_ms::Real,
            bin_size_ms::Real;
            rng::AbstractRNG=Random.GLOBAL_RNG
        ) -> Vector{Bool}

    Generate a binary Poisson spike train.

    Each time bin of width `bin_size_ms` fires independently with probability
    `rate_hz * bin_size_ms / 1000`. Returns a `Bool` vector of length
    `round(Int, duration_ms / bin_size_ms)`.
    """
    function poisson_spike_train(
        rate_hz::Real,
        duration_ms::Real,
        bin_size_ms::Real;
        rng::AbstractRNG=Random.GLOBAL_RNG,
    )
        if rate_hz <= 0
            throw(ArgumentError("rate_hz must be positive, got $rate_hz"))
        end
        if duration_ms <= 0
            throw(ArgumentError("duration_ms must be positive, got $duration_ms"))
        end
        if bin_size_ms <= 0
            throw(ArgumentError("bin_size_ms must be positive, got $bin_size_ms"))
        end
        n_bins = round(Int, duration_ms / bin_size_ms)
        probability = rate_hz * bin_size_ms / 1000.0
        return rand(rng, n_bins) .< probability
    end

    demo_rng = MersenneTwister(seed_demo)
    demo_train = poisson_spike_train(lambda_demo, duration_demo, 1.0; rng=demo_rng)
    demo_times = (1:length(demo_train))[demo_train]

    p_spike = scatter(
        demo_times,
        ones(length(demo_times));
        marker=(:vline, 8, :black),
        legend=false,
        xlabel="Time (ms)",
        title="Spike Train  |  λ = $(lambda_demo) Hz  |  $(sum(demo_train)) spikes",
        yticks=false,
        ylims=(0.5, 1.5),
        framestyle=:box,
        size=(700, 150),
    )
end

# ╔═╡ 11111111-0000-0000-0000-000000000005
md"""
---
## 📊 Part 2 — The Cross-Correlogram (CCG)

The **cross-correlogram** counts how often neuron B fires at a lag τ relative to
each spike in neuron A. It is built by:

1. For every spike in neuron A (the **reference** neuron), mark time 0.
2. Count spikes from neuron B in each time bin relative to that reference.
3. Sum across all reference spikes.

A peak near **τ = 0** suggests synchronous firing.
A peak at **τ > 0** suggests A tends to precede B (possible excitatory connection).
A **trough** suggests inhibition.

> 💡 Try adjusting the coupling delay below — notice how the CCG peak shifts!
"""

# ╔═╡ 11111111-0000-0000-0000-000000000006
md"""
**Neuron A rate (λ_A):** $(@bind rate_a Slider(5:5:60; default=20, show_value=true)) Hz

**Neuron B rate (λ_B):** $(@bind rate_b Slider(5:5:60; default=20, show_value=true)) Hz

**Coupling delay (ms):** $(@bind coupling_delay Slider(0:1:30; default=5, show_value=true)) ms

**Coupling strength:** $(@bind coupling_strength Slider(0.0:0.05:1.0; default=0.3, show_value=true))

**Number of trials:** $(@bind n_trials_ccg Slider(5:5:100; default=30, show_value=true))

**Max lag (ms):** $(@bind max_lag Slider(20:10:200; default=80, show_value=true)) ms
"""

# ╔═╡ 11111111-0000-0000-0000-000000000007
begin
    """
        generate_coupled_pair(
            rate_a_hz::Real,
            rate_b_hz::Real,
            duration_ms::Real,
            delay_ms::Int,
            strength::Real;
            rng::AbstractRNG=Random.GLOBAL_RNG
        ) -> Tuple{Vector{Bool}, Vector{Bool}}

    Simulate a pair of spike trains where spikes in A can drive spikes in B
    after `delay_ms` milliseconds with probability `strength`.

    Returns `(train_a, train_b)` as binary vectors.
    """
    function generate_coupled_pair(
        rate_a_hz::Real,
        rate_b_hz::Real,
        duration_ms::Real,
        delay_ms::Int,
        strength::Real;
        rng::AbstractRNG=Random.GLOBAL_RNG,
    )
        if !(0.0 <= strength <= 1.0)
            throw(ArgumentError("strength must be in [0, 1], got $strength"))
        end
        train_a = poisson_spike_train(rate_a_hz, duration_ms, 1.0; rng=rng)
        train_b = poisson_spike_train(rate_b_hz, duration_ms, 1.0; rng=rng)
        n = length(train_a)
        for t in 1:(n - delay_ms)
            if train_a[t] && rand(rng) < strength
                target = t + delay_ms
                if target <= n
                    train_b[target] = true
                end
            end
        end
        return (train_a, train_b)
    end

    """
        compute_ccg(
            train_a::AbstractVector{Bool},
            train_b::AbstractVector{Bool},
            max_lag::Int
        ) -> Vector{Int}

    Compute the raw cross-correlogram between two binary spike trains.

    For each spike in `train_a`, counts spikes in `train_b` at lags
    -`max_lag` to +`max_lag` (in bins). Returns a count vector of length
    `2 * max_lag + 1`.
    """
    function compute_ccg(
        train_a::AbstractVector{Bool},
        train_b::AbstractVector{Bool},
        max_lag_bins::Int,
    )
        if length(train_a) != length(train_b)
            throw(
                ArgumentError(
                    "train_a and train_b must have equal length, got " *
                    "$(length(train_a)) and $(length(train_b))",
                ),
            )
        end
        n = length(train_a)
        lags = -max_lag_bins:max_lag_bins
        counts = zeros(Int, length(lags))
        spike_times_a = findall(train_a)
        spike_times_b = findall(train_b)
        for t_a in spike_times_a
            for (lag_index, lag) in enumerate(lags)
                t_b = t_a + lag
                if 1 <= t_b <= n && train_b[t_b]
                    counts[lag_index] += 1
                end
            end
        end
        return counts
    end

    trial_duration_ms = 500
    ccg_rng = MersenneTwister(42)

    pairs = [
        generate_coupled_pair(
            rate_a,
            rate_b,
            trial_duration_ms,
            coupling_delay,
            coupling_strength;
            rng=ccg_rng,
        ) for _ in 1:n_trials_ccg
    ]

    raw_ccg = reduce(
        (accumulator, pair) ->
            accumulator .+ compute_ccg(pair[1], pair[2], max_lag),
        pairs;
        init=zeros(Int, 2 * max_lag + 1),
    )

    lag_axis = (-max_lag):max_lag

    p_ccg = bar(
        lag_axis,
        raw_ccg;
        xlabel="Lag τ (ms)",
        ylabel="Spike count",
        title="Raw Cross-Correlogram  |  delay=$(coupling_delay)ms  strength=$(coupling_strength)",
        legend=false,
        color=:steelblue,
        linecolor=:steelblue,
        size=(700, 300),
    )
    vline!(p_ccg, [0]; color=:red, linestyle=:dash, linewidth=1.5, label="τ = 0")
end

# ╔═╡ 11111111-0000-0000-0000-000000000008
md"""
---
## ⚠️ Part 3 — The Common Input Problem

Imagine both neurons receive input from the **same upstream stimulus** (e.g., a visual
grating appearing on each trial). They will both respond at roughly the same time on
every trial — making the CCG look as if they are directly connected, even if they
share no synapse.

This is the **common input artifact**. Below, we simulate it: both neurons respond to
a shared stimulus, but have **no direct connection**.

> 💡 Set coupling strength = 0 above and observe that the raw CCG still has a peak
>    when a shared stimulus drives both neurons.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000009
md"""
**Stimulus-evoked rate boost:** $(@bind stim_boost Slider(0:5:80; default=30, show_value=true)) Hz

**Stimulus window (ms):** $(@bind stim_window Slider(10:10:200; default=50, show_value=true)) ms

**Stimulus onset (ms):** $(@bind stim_onset Slider(50:10:300; default=150, show_value=true)) ms

**Trials for common-input demo:** $(@bind n_trials_stim Slider(10:10:150; default=60, show_value=true))
"""

# ╔═╡ 11111111-0000-0000-0000-000000000010
begin
    """
        generate_stimulus_driven_pair(
            baseline_rate_hz::Real,
            stim_rate_boost_hz::Real,
            duration_ms::Real,
            stim_onset_ms::Int,
            stim_window_ms::Int;
            rng::AbstractRNG=Random.GLOBAL_RNG
        ) -> Tuple{Vector{Bool}, Vector{Bool}}

    Simulate two independent neurons driven by a common stimulus.

    Both neurons fire at `baseline_rate_hz` outside the stimulus window, and at
    `baseline_rate_hz + stim_rate_boost_hz` during the stimulus. The two trains
    are otherwise independent — no direct coupling.
    """
    function generate_stimulus_driven_pair(
        baseline_rate_hz::Real,
        stim_rate_boost_hz::Real,
        duration_ms::Real,
        stim_onset_ms::Int,
        stim_window_ms::Int;
        rng::AbstractRNG=Random.GLOBAL_RNG,
    )
        n_bins = round(Int, duration_ms)
        stim_end = stim_onset_ms + stim_window_ms

        function make_train(r)
            train = falses(n_bins)
            for t in 1:n_bins
                rate = (stim_onset_ms <= t < stim_end) ? r + stim_rate_boost_hz : r
                probability = clamp(rate / 1000.0, 0.0, 1.0)
                train[t] = rand(rng) < probability
            end
            return train
        end

        return (make_train(baseline_rate_hz), make_train(baseline_rate_hz))
    end

    stim_rng = MersenneTwister(7)
    stim_max_lag = 80

    stim_pairs = [
        generate_stimulus_driven_pair(
            10.0,
            Float64(stim_boost),
            500.0,
            stim_onset,
            stim_window;
            rng=stim_rng,
        ) for _ in 1:n_trials_stim
    ]

    stim_ccg = reduce(
        (accumulator, pair) ->
            accumulator .+ compute_ccg(pair[1], pair[2], stim_max_lag),
        stim_pairs;
        init=zeros(Int, 2 * stim_max_lag + 1),
    )

    stim_lag_axis = (-stim_max_lag):stim_max_lag

    p_stim = bar(
        stim_lag_axis,
        stim_ccg;
        xlabel="Lag τ (ms)",
        ylabel="Spike count",
        title="CCG with Common Stimulus Input (no direct coupling)",
        legend=false,
        color=:darkorange,
        linecolor=:darkorange,
        size=(700, 300),
    )
    vline!(p_stim, [0]; color=:red, linestyle=:dash, linewidth=1.5)
end

# ╔═╡ 11111111-0000-0000-0000-000000000011
md"""
> The broad peak centered at τ = 0 reflects the shared stimulus, not a synapse.
> This is exactly what the **shift predictor** is designed to remove.

---
## 🔄 Part 4 — The Shift Predictor

The **shift predictor** estimates the stimulus-driven baseline by correlating spikes
from neuron A on trial *n* with spikes from neuron B on trial *n+1* (a cyclic shift).

- The shifted pairs share the same stimulus statistics.
- But the within-trial temporal structure (synaptic interactions) is destroyed.
- Subtracting the shift predictor from the raw CCG isolates true network correlations.

```
Corrected CCG = Raw CCG − Shift Predictor
```
"""

# ╔═╡ 11111111-0000-0000-0000-000000000012
md"""
**Coupling delay (ms):** $(@bind sp_delay Slider(0:1:30; default=8, show_value=true)) ms

**Coupling strength:** $(@bind sp_strength Slider(0.0:0.05:1.0; default=0.4, show_value=true))

**Stimulus boost (Hz):** $(@bind sp_stim_boost Slider(0:5:80; default=40, show_value=true))

**Number of trials:** $(@bind sp_n_trials Slider(20:10:200; default=80, show_value=true))

**Max lag (ms):** $(@bind sp_max_lag Slider(20:10:150; default=60, show_value=true)) ms
"""

# ╔═╡ 11111111-0000-0000-0000-000000000013
begin
    """
        compute_shift_predictor(
            trains_a::AbstractVector{<:AbstractVector{Bool}},
            trains_b::AbstractVector{<:AbstractVector{Bool}},
            max_lag_bins::Int
        ) -> Vector{Int}

    Compute the shift predictor by correlating train A from trial n with
    train B from trial n+1 (cyclic), then averaging over all shifts.

    Returns a count vector of length `2 * max_lag_bins + 1`.
    """
    function compute_shift_predictor(
        trains_a::AbstractVector{<:AbstractVector{Bool}},
        trains_b::AbstractVector{<:AbstractVector{Bool}},
        max_lag_bins::Int,
    )
        if length(trains_a) != length(trains_b)
            throw(
                ArgumentError(
                    "trains_a and trains_b must have the same number of trials",
                ),
            )
        end
        n_trials = length(trains_a)
        accumulator = zeros(Int, 2 * max_lag_bins + 1)
        for index in 1:n_trials
            shifted_index = mod1(index + 1, n_trials)
            accumulator .+= compute_ccg(
                trains_a[index], trains_b[shifted_index], max_lag_bins
            )
        end
        return accumulator
    end

    sp_rng = MersenneTwister(13)
    stim_onset_sp = 150
    stim_window_sp = 60

    function generate_full_pair(rng)
        n_bins = 500
        stim_end = stim_onset_sp + stim_window_sp
        function make_train_with_rate(base_rate)
            train = falses(n_bins)
            for t in 1:n_bins
                rate = (stim_onset_sp <= t < stim_end) ?
                    base_rate + sp_stim_boost : base_rate
                train[t] = rand(rng) < clamp(rate / 1000.0, 0.0, 1.0)
            end
            return train
        end
        train_a = make_train_with_rate(15.0)
        train_b = make_train_with_rate(15.0)
        # add coupling from A to B
        for t in 1:(n_bins - sp_delay)
            if train_a[t] && rand(rng) < sp_strength
                train_b[t + sp_delay] = true
            end
        end
        return (train_a, train_b)
    end

    all_pairs = [generate_full_pair(sp_rng) for _ in 1:sp_n_trials]
    trains_a_all = map(first, all_pairs)
    trains_b_all = map(last, all_pairs)

    sp_raw = reduce(
        (accumulator, pair) ->
            accumulator .+ compute_ccg(pair[1], pair[2], sp_max_lag),
        all_pairs;
        init=zeros(Int, 2 * sp_max_lag + 1),
    )

    sp_predictor = compute_shift_predictor(trains_a_all, trains_b_all, sp_max_lag)
    sp_corrected = sp_raw .- sp_predictor

    sp_lag_axis = (-sp_max_lag):sp_max_lag

    p_raw_sp = bar(
        sp_lag_axis,
        sp_raw;
        label="Raw CCG",
        color=:steelblue,
        linecolor=:steelblue,
        alpha=0.7,
        xlabel="Lag τ (ms)",
        ylabel="Count",
        title="Raw CCG vs Shift Predictor",
    )
    bar!(
        p_raw_sp,
        sp_lag_axis,
        sp_predictor;
        label="Shift Predictor",
        color=:darkorange,
        linecolor=:darkorange,
        alpha=0.6,
    )
    vline!(p_raw_sp, [0]; color=:red, linestyle=:dash, linewidth=1.5, label="τ = 0")

    p_corrected = bar(
        sp_lag_axis,
        sp_corrected;
        label="Corrected CCG",
        color=:green,
        linecolor=:green,
        alpha=0.8,
        xlabel="Lag τ (ms)",
        ylabel="Count",
        title="Corrected CCG  (Raw − Shift Predictor)  |  coupling delay = $(sp_delay) ms",
    )
    vline!(p_corrected, [0]; color=:red, linestyle=:dash, linewidth=1.5, label="τ = 0")
    vline!(
        p_corrected,
        [sp_delay];
        color=:purple,
        linestyle=:dot,
        linewidth=2,
        label="Expected peak ($(sp_delay) ms)",
    )

    plot(p_raw_sp, p_corrected; layout=(2, 1), size=(700, 500))
end

# ╔═╡ 11111111-0000-0000-0000-000000000014
md"""
> 🟢 The **corrected CCG** (bottom panel) shows a peak at exactly the coupling delay you
> set — with the broad stimulus-driven baseline removed.
> 🟠 The **shift predictor** (top panel, orange) captures what the CCG would look like
> with only common-stimulus correlation and no direct coupling.

---
## 📐 Part 5 — Raster Plot Explorer

Below you can inspect the trial-by-trial raster plots for both neurons side by side.
Stimulus window is shown in shading. Notice how both neurons fire together during the
stimulus — this is the source of the common-input artifact.

**Number of trials to display:** $(@bind n_raster Slider(5:5:50; default=20, show_value=true))
"""

# ╔═╡ 11111111-0000-0000-0000-000000000015
begin
    raster_rng = MersenneTwister(99)
    raster_pairs = [generate_full_pair(raster_rng) for _ in 1:n_raster]

    p_raster_a = plot(;
        xlabel="Time (ms)",
        ylabel="Trial",
        title="Neuron A",
        legend=false,
        size=(700, 400),
        ylims=(0, n_raster + 1),
    )
    p_raster_b = plot(;
        xlabel="Time (ms)",
        ylabel="Trial",
        title="Neuron B",
        legend=false,
        size=(700, 400),
        ylims=(0, n_raster + 1),
    )

    for (trial_index, (train_a_trial, train_b_trial)) in enumerate(raster_pairs)
        spike_times_a_trial = findall(train_a_trial)
        spike_times_b_trial = findall(train_b_trial)
        scatter!(
            p_raster_a,
            spike_times_a_trial,
            fill(trial_index, length(spike_times_a_trial));
            marker=(:vline, 6, :steelblue),
            markeralpha=0.8,
        )
        scatter!(
            p_raster_b,
            spike_times_b_trial,
            fill(trial_index, length(spike_times_b_trial));
            marker=(:vline, 6, :crimson),
            markeralpha=0.8,
        )
    end

    vspan!(
        p_raster_a,
        [stim_onset_sp, stim_onset_sp + stim_window_sp];
        alpha=0.15,
        color=:gold,
        label="Stimulus",
    )
    vspan!(
        p_raster_b,
        [stim_onset_sp, stim_onset_sp + stim_window_sp];
        alpha=0.15,
        color=:gold,
        label="Stimulus",
    )

    plot(p_raster_a, p_raster_b; layout=(1, 2), size=(900, 350))
end

# ╔═╡ 11111111-0000-0000-0000-000000000016
md"""
---
## 🧩 Part 6 — Interpreting CCG Shapes

Different CCG patterns carry distinct physiological interpretations:

| CCG Pattern | Interpretation |
|---|---|
| Sharp peak at τ = 0 | Common input or synchronization |
| Asymmetric peak at τ > 0 | A excites B with a delay |
| Asymmetric peak at τ < 0 | B excites A with a delay |
| Trough at τ = 0 | Mutual inhibition |
| Broad symmetric peak | Shared oscillatory input |
| Flat after correction | No direct interaction |

> ✅ After shift predictor correction, a **sharp peak at the coupling delay** is the
> gold standard evidence for a **monosynaptic excitatory connection**.

---
## 📚 Summary

| Concept | Key Idea |
|---|---|
| **Spike train** | Binary time series of neural firing |
| **Cross-correlogram** | Counts co-firing at each lag τ |
| **Common input artifact** | Shared stimulus inflates CCG artificially |
| **Shift predictor** | Correlates A(trial n) × B(trial n+1) to estimate stimulus baseline |
| **Corrected CCG** | Raw CCG − Shift Predictor = true network interactions |

### Next steps to explore
- Try setting **coupling strength = 0** and **stimulus boost > 0** — the shift predictor
  should fully explain the raw CCG.
- Try **coupling strength > 0** and **stimulus boost = 0** — the raw CCG and corrected
  CCG should look similar (no artifact to remove).
- Increase **number of trials** to see the estimates stabilize.

---
*Built with Julia · Pluto.jl · Plots.jl*
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
begin
    # Notebook-level package manifest (Pluto auto-manages this)
    # Required packages: PlutoUI, Plots, Random, Statistics, StatsBase
    nothing
end
