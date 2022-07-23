# import packages needed
using Catalyst, DifferentialEquations, Plots, DataFrames, GLM, Statistics, Distributions, CSV, GraphViz
using DifferentialEquations.EnsembleAnalysis


## simple reaction network model
coreMDL = @reaction_network begin
    (ρₘ,dₘ), ∅ <--> mRNA          #transcription and degradation of mRNA
    kₚ, mRNA --> mRNA + protein   #translation of mRNA to protein
    dₚ, protein --> ∅             #protein degradation
end ρₘ dₘ kₚ dₚ
##


## extend to sequestration model
sqstMDL = @reaction_network begin
    (ρₘ,dₘ), ∅ <--> mRNA                   #transcription and degradation of mRNA
    (ρₛ, dₛ), ∅ <--> sRNA;                  #transcription and degradation of sRNA
    (kₛₘ, rₛₘ), mRNA + sRNA <--> smRNA      #formation of sRNA-mRNA complex
    dₖₘ, smRNA --> sRNA                     #catalytic degradation of mRNA by sRNA-mRNA complex
    kₚ, mRNA --> mRNA + protein             #translation of mRNA to protein
    dₚ, protein --> ∅                       #protein degradation
end ρₘ dₘ ρₛ dₛ kₛₘ rₛₘ dₖₘ kₚ dₚ
#g = complexgraph(sqstMDL)
##


vec1 = 0.01 : 0.01 : 0.1
vec2 = 0.2 : 0.1 : 1.0
vec3 = 2.0 : 1.0 : 10.0
logRange = [vec1 ; vec2; vec3] # create a logscale range for parasweep across diff orders of magnitude


# chemical species index: 1 - mRNA, 2 - sRNA, 3 - smRNA, 4 - protein 
function singleSweep(reactionNetwork, u0, tspan, p0, pRange, trajectNum, paraIndex, chemSpecIndex)
    
    columnNames = [string.(reactionparams(reactionNetwork)) ; ["mean", "variance", "fano_factor", "tspan", "trajectories", "stationary"]]
    df = DataFrame([[] for _ = columnNames] , columnNames) # construct dataframe to hold all parameter values and statistics
    
    for i in pRange # for loop to iterate over parameter values
           
        p0[paraIndex] = i; # change parameter that is currently being swept over

        # Run Simulation
        simulation = runSim(reactionNetwork, u0, tspan, p0, trajectNum)
        
        #display(plot(simulation, linealpha = 1, linewidth = 1))
        #break
        # check for stationary distribution, store results in dataframes
        push!(df, stationaryDist(simulation, chemSpecIndex, tspan, p0, trajectNum));
    end
    
    #print(df)
    return df;
end

# chemSpecIndex corresponds to chemical species of interest: 1 - mRNA, 2 - sRNA, 3 - smRNA, 4 - protein
function stationaryDist(simulation, chemSpecIndex, tspan, pValues, trajectNum)
    ans = ["Yes", "No"];
    ts = tspan[1] : 1 : tspan[end];

    m_series,v_series = timeseries_point_meanvar(simulation,ts) # obtain mean and variance at each time point
    
    #plot(ts, m_series[chemSpecIndex,:], labels="Mean")
    #display(plot!(ts, v_series[chemSpecIndex,:], title=pValues, labels="Variance")) 

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
    return [pValues ;[mean_stat, var_stat, var_stat/mean_stat, ts[end], trajectNum, stationary]]
end

function generateFig(unregMDL, sequestMDL, p0_1, p0_2, para1, para2, paraNonsweepValues, tspan, trajectNum)
    # find index of parameter strings passed into function
    paraIndex1 = findfirst(==(para1), string.(reactionparams(sequestMDL)))
    paraIndex2 = findfirst(==(para2), string.(reactionparams(sequestMDL)))


    # Run simulation with unregulated model
    unregSim = runSim(unregMDL, [0., 0.,], tspan, p0_1, trajectNum)
    unregResults = stationaryDist(unregSim, 2, tspan, p0_1, trajectNum)
    ff_mean_p0 = [unregResults[5], unregResults[7]]; # mean and fano-factor, in that order, in a vector
    hline([ff_mean_p0[2]],labels="Poisson Noise", xaxis=:log)
    #plot horizontal line showing a fano factor of 1 in the sequestration model
    for i in paraNonsweepValues
        p0_2[paraIndex1] = i
        df = singleSweep(sequestMDL, [0., 0., 0., 0.], tspan, p0_2, logRange, trajectNum, paraIndex2, 4)
        label = para1 * "=" * string.(i)
        plot!(ff_mean_p0[1] ./ df[:, :mean], ff_mean_p0[2] ./ df[:, :fano_factor], labels=label, lw = 2, xaxis=:log, legend= :outertopleft)
    end
    display(plot!(bg=:white, xaxis=:log);)
    #plot deterministic solution
end

function runSim(network, u0, tspan, p0, trajectNum)
    # Run Simulation
    @named odeSys = convert(ODESystem, network); # create ODE Problem
    prob = ODEProblem(odeSys,u0,tspan,p0;jac=true,sparse=true);

    ssprob = SteadyStateProblem(prob); # convert to Steady State Problem
    ss_sol = round.(solve(ssprob, SSRootfind())); # store steady state solutions
    dprob = DiscreteProblem(network, ss_sol, tspan, p0); # create stochastic discrete and jump problems (gillespie) with tspan input
    jprob = JumpProblem(network, dprob, Direct());

    rand_u0 = zeros(length(ss_sol))
    for a in 1 : length(ss_sol)
        rand_u0[a] = rand(Poisson(ss_sol[a]))
    end

    function prob_func(prob,i,repeat)
        @. prob.prob.u0 = copy(rand_u0); # draw initial conditions for discrete prob from poisson dist 
        prob
    end

    ensemble_prob = EnsembleProblem(jprob, prob_func=prob_func); # solve using ensemble problem 
    simulation = solve(ensemble_prob, SSAStepper(), EnsembleThreads(), trajectories=trajectNum);

    return simulation
    
end


#generating fig 1: increasing catalytic degradation
generateFig(coreMDL, sqstMDL, [1.0, 0.1, 1.0, 0.01], [1.0, 0.1, 1.0, 0.1, 1.0, 0.01, 0, 1.0, 0.01], "dₖₘ", "ρₛ", [0, 1e-5, 1e-4, 1e-3, 1e-2], (0., 50.), 6)

# generating fig 2: decreasing the sequestration effect
#generateFig(coreMDL, sqstMDL, [1.0, 0.1, 1.0, 0.01], [1.0, 0.1, 1.0, 0.1, 1.0, 0.01, 1e-2, 1.0, 0.01], 6, 3, [1e-2, 1e-1, 1e0, 1e1], (0., 1000.), 90)
# generating fig 3: high sRNA translation, no sequestration, increasing catalytic degradation
#generateFig(coreMDL, sqstMDL, [1.0, 0.1, 1.0, 0.01], [1.0, 0.1, 1.0, 0.1, 1.0, 100, 0, 1.0, 0.01], 7, 3, [0.01, 0.05, 0.1, 1], (0., 1000.), 90)



#=
# code for generating protein and mRNA level figures
sqstSim = runSim(sqstMDL, [0., 0., 0., 0.], (0., 50.),[1.0, 0.1, 1.0, 0.1, 1.0, 0.01, 0, 1.0, 0.01], 1)
coreSim = runSim(coreMDL, [0., 0.], (0., 50.),[1.0, 0.1, 1.0, 0.01], 1)

ts = 0 : 1 : 25;
m_series,v_series = timeseries_point_meanvar(sqstSim,ts) 
m_series1,v_series1 = timeseries_point_meanvar(coreSim,ts) 
l = @layout [a ; b]
p1 = plot(coreSim, vars=(0, 2), labels="kₛₘ=0", linewidth=2.0, legend = :outertopleft)
p1 = plot!(sqstSim, vars=(0, 4),labels="kₛₘ = 1.0",linewidth=2.0, legend = :outertopleft)
p2 = plot(coreSim, vars=(0, 1), labels="kₛₘ=0",linewidth=2.0, legend = :outertopleft)
p2 = plot!(sqstSim, vars=(0, 1), labels="kₛₘ = 1.0",linewidth=2.0, legend = :outertopleft)

plot(p1, p2, layout = l)

plot(ts, m_series1[2, :], labels="kₛₘ=0", linewidth=2.5)
display(plot!(ts, m_series[4,:], labels="kₛₘ = 1.0",linewidth=2.5))

plot(ts, m_series1[1, :], labels="kₛₘ=0",linewidth=2.5)
display(plot!(ts, m_series[1,:], labels="kₛₘ = 1.0",linewidth=2.5))
=#

#=
# perform a parameter sweep on all parameters in sequestration MDL
df = singleSweep(sqstMDL, [0.,0., 0., 0.], (0.,750.), [1.0, 0.1, 1.0, 0.1, 1.0, 0.01, 1.0, 0.01], logRange, 36, 4)
CSV.write("//Users//merylliu//Desktop//SURP 2022//noiseModel//"*"parasweep_sqst_protein.csv", df)
counter = 0;
for i in 1 : length(string.(reactionparams(sqstMDL)))
    global counter
    counter +=28
    display(plot(df[(counter-27):counter, i], df[(counter-27):counter, :fano_factor], seriestype = :scatter, title=string.(reactionparams(sqstMDL))[i]*" vs. Fano Factor, sqstMDL", xlabel=string.(reactionparams(sqstMDL))[i], ylabel="Fano-Factor",legend = false))
    
end

# compare theoretical and simulation results for transcription of sRNA, ρₛ
plot(df[57:84, :ρₛ], df[57:84, :fano_factor], seriestype = :scatter, title="Theoretcial vs. Simulation, ρₛ, sqstMDL", xlabel="ρₛ", ylabel="Fano-Factor",labels="Simulation")
ρₛ_variable = 0 : 0.01 : 10.0;
display(plot!(ρₛ_variable, 1 .+ (10) .* (1.0 ./ (1.0 .+ ρₛ_variable)), labels="Theoretical"))
##

# parameter sweep on unregulated translation coreMDL
df_2 = singleSweep(coreMDL, [0., 0.], (0., 750.), [1.0, 0.1, 1.0, 0.01], logRange, 36, 2)
CSV.write("//Users//merylliu//Desktop//SURP 2022//noiseModel//"*"parasweep_core_protein.csv", df)
counter = 0;
for i in 1 : length(string.(reactionparams(coreMDL)))
    global counter
    counter +=28
    display(plot(df_2[(counter-27):counter, i], df_2[(counter-27):counter, :fano_factor], seriestype = :scatter, title=string.(reactionparams(coreMDL))[i]*" vs. Fano Factor, coreMDL", xlabel=string.(reactionparams(sqstMDL))[i], ylabel="Fano-Factor",legend = false))
end
# compare theoretical and simulation results for burst factor vs. fano factor
plot(df_2[29:56, :dₘ], df_2[29:56, :fano_factor], seriestype = :scatter, title="Theoretical vs. Simulation, dₘ, coreMDL", xlabel="dₘ", ylabel="Fano-Factor",labels="Simulation")
dₘ_variable = 0 : 0.01 : 10.0
display(plot!(dₘ_variable, 1 .+ (1.0 ./ dₘ_variable), labels="Theoretical"))

=#
