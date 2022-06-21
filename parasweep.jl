# import packages needed
using Catalyst, DifferentialEquations, Plots, DataFrames, GLM, Statistics, Distributions, CSV
using DifferentialEquations.EnsembleAnalysis

# create reaction network model
coreMDL = @reaction_network begin
    (ρₘ,dₘ), ∅ <--> mRNA          #transcription and degradation of mRNA
    kₚ, mRNA --> mRNA + protein   #translation of mRNA to protein
    dₚ, protein --> ∅             #protein degradation
end ρₘ dₘ kₚ dₚ

# parameter index: 1 - ρₘ, 2 - dₘ, 3 - kₚ, 4 - dₚ, chemical species index: 1 - mRNA, 2 - protein
function singleSweep(reactionNetwork, u0, paraIndex, tspan, p, trajectNum, chemSpecIndex)
  vec1 = 0.01 : 0.01 : 0.1
  vec2 = 0.2 : 0.1 : 1.0
  para = ["ρₘ", "dₘ", "kₚ", "dₚ"];
  spec = ["mRNA", "protein"]
  logscale = [vec1 ; vec2] # create a log scale across orders of magnitude

  # construct empty Dataframe
  df = DataFrame(Parameter=String[], ParameterValue = Float64[], Mean=Float64[], Variance=Float64[], FanoFactor=Float64[], Stationary = String[]);
   
  for i in logscale # for loop to iterate over parameter values
    p[paraIndex] = i; # change parameter

    # Run Simulation
    @named odeSys = convert(ODESystem,reactionNetwork); # create ODE Problem
    prob = ODEProblem(odeSys,u0,tspan,p;jac=true,sparse=true);
    ssprob = SteadyStateProblem(prob); # convert to Steady State Problem
    ss_sol = round.(solve(ssprob, SSRootfind())); # store steady state solutions
    dprob = DiscreteProblem(coreMDL, ss_sol, tspan, p); # create stochastic discrete and jump problems (gillespie) with tspan input
    jprob = JumpProblem(coreMDL, dprob, Direct());

    function prob_func(prob,i,repeat)
        @. prob.prob.u0 = [rand(Poisson(ss_sol[1])), rand(Poisson(ss_sol[2]))]; # draw initial conditions for discrete prob from poisson dist 
        prob
    end

    ensemble_prob = EnsembleProblem(jprob, prob_func=prob_func); # solve using ensemble problem 
    simulation = solve(ensemble_prob, SSAStepper(), EnsembleThreads(), trajectories=trajectNum);
    #display(plot(simulation, linealpha = 0.5, linewidth = 1))

    # check for stationary distribution, store results in dataframes
    push!(df, stationaryDist(simulation, chemSpecIndex, tspan, paraIndex, i));
   end

   #print(df)
   display(plot(df[:, :ParameterValue], df[:, :FanoFactor], seriestype = :scatter, title=para[paraIndex]*" vs. Fano Factor", xlabel=para[paraIndex], ylabel="Fano-Factor",legend = false))
   CSV.write("//Users//merylliu//Desktop//SURP 2022//noiseModel//"*para[paraIndex]*"_"*spec[chemSpecIndex]*"_sweep.csv", df);
end

# chemSpecIndex corresponds to chemical species of interest: 1 - mRNA, 2 - protein
function stationaryDist(simulation, chemSpecIndex, tspan, paraIndex, paraValue)
    ans = ["Yes", "No"];
    para = ["ρₘ", "dₘ", "kₚ", "dₚ"];
    ts = tspan[1] : 1 : tspan[end];

    m_series,v_series = timeseries_point_meanvar(simulation,ts) # obtain mean and variance at each time point
    
    #plot(ts, m_series[chemSpecIndex,:], labels="Mean")
    #display(plot!(ts, v_series[chemSpecIndex,:], title=para[paraIndex]*"="*string(paraValue), labels="Variance")) 

    # Linear regression to check for stationary distribution
    recent_cutoff = .20; 
    slope_threshold = 1000; # cutoff for stationary slope
    
    # obtain the x% most recent time points
    spec_mean = m_series[chemSpecIndex, :]
    spec_var = v_series[chemSpecIndex, :]
    mean_recent = spec_mean[Int(round(tspan[end] - recent_cutoff * tspan[end])):Int(tspan[end])]
    var_recent = spec_var[Int(round(tspan[end] - recent_cutoff * tspan[end])):Int(tspan[end])]
    mean_recent = mean_recent .- mean(mean_recent); # normalize data around 1
    var_recent = var_recent .- mean(var_recent);
   
    # create dataframes + conduct linear regression
    data_mSeries = DataFrame(X = 0 : 1 : length(mean_recent) - 1, Y = mean_recent)
    data_vSeries = DataFrame(X = 0 : 1 : length(var_recent) - 1, Y = var_recent)
    linMean = lm(@formula(Y ~ X), data_mSeries);
    linVar = lm(@formula(Y ~ X), data_vSeries);

    # store coefficients in tuple type? index 2 = slope
    mean_coef = coef(linMean);
    var_coef = coef(linVar);
    #println(mean_coef)
    #println(var_coef)

    if (abs(mean_coef[2]) > slope_threshold || abs(var_coef[2]) > slope_threshold) # check if distribution is stationary
        stationary = ans[2];
    else
        stationary = ans[1];
    end

    # plot linear regression for visualization
    #x = 0 : 1 : length(mean_recent) - 1
    #mean_lm(x) = mean_coef[1] + mean_coef[2] * x
    #var_lm(x) = var_coef[1] + var_coef[2] * x
    #display(plot(x, mean_lm.(x), legend=false))
    #display(plot(x, var_lm.(x), legend=false))

    # obtain mean and variance from the end of simulation
    m,v = timepoint_meanvar(simulation, ts[end])
    mean_stat = m[chemSpecIndex];
    var_stat = v[chemSpecIndex];

    # store parameter, parameter value, mean, variance, FF, stability of simulation
    return (para[paraIndex], paraValue, mean_stat, var_stat, var_stat/mean_stat, stationary)
end


#%% test code

# perform a parameter sweep on p_m
singleSweep(coreMDL, [0.,0.], 1, (0.,1500.), [1.0, 0.1, 1.0, 0.01], 180, 2)
# single parameter sweep on d_m
singleSweep(coreMDL, [0.,0.], 2, (0.,1500.), [1.0, 0.1, 1.0, 0.01], 180, 2)
# single parameter sweep on k_p
singleSweep(coreMDL, [0.,0.], 3, (0.,1500.), [1.0, 0.1, 1.0, 0.01], 180, 2)
# single parameter sweep on d_m
singleSweep(coreMDL, [0.,0.], 4, (0.,1500.), [1.0, 0.1, 1.0, 0.01], 180, 2)

#%%
