# main_cstr
using JuMP, Ipopt, LinearAlgebra, Gurobi, Random, Plots, DataFrames, CSV, Printf, Statistics
include("fop.jl")
include("surrogate_fop.jl")
include("init_iop.jl")
include("iop_subproblems.jl")
include("bcd_main.jl")

rng = MersenneTwister(1234)

# Process Parameters
tau = 60 # s
C1 = 5e3 # 1/s
Cm1 = 1e6 # 1/s
Q = 10000 # cal/mol
Qm1 = 15000 # cal/mol
gc = 1.987 # cal/mol.K
dH = -5000 # cal/mol
rho = 1.0 # kg/L
Cp = 1000 # cal/kg.K
flag = 0

# Steady State Values
A_ss = 0.494
R_ss = 0.505 # Controlled Variables
T_ss = 429.53

Ti_ss = 427 # Manipulated Variable
Ai_ss = 1.0
Ri_ss = 0.0

Err_evol_1 = []
Samples = [5, 10, 25, 50, 75, 100, 250, 500, 1000]
# Samples = [1000]

Result1 = zeros(Float64, 10, length(Samples))
Result2 = zeros(Float64, 10, length(Samples))
Ai_test = [Ai_ss + rand(rng, -80:100) / 100 for _ = 1:100]

# temperature_profiles = zeros(Float64, 10, length(Ai_test))
done = false
ro = 0
while !done
    global ro
    ro += 1
    if ro == 10
        global done = true
    end
    I = maximum(Samples) # Size of training data
    global Ai = [Ai_ss + rand(rng, -80:100) / 100 for _ = 1:I]
    global Ti_train = zeros(Float64, I)
    global A0_train = zeros(Float64, I)
    global R0_train = zeros(Float64, I)
    global T0_train = zeros(Float64, I)

    # Generate random training data by introducing disturbances
    for i = 1:I
        Ti_train[i], A0_train[i], R0_train[i], T0_train[i] = RTO(Ai[i], Ri_ss)
    end

    colum = 1
    for I in Samples
        local Qc, P, train_loss = penaltyIOP(I, Ai[1:I], Ri_ss, 1)
        if flag == 1
            ro = ro - 1
            global flag = 0
            break
        end
        # Validation Phase (fixed amount of data)
        I = 100

        Ti_test = zeros(Float64, I)
        A0_test = zeros(Float64, I)
        R0_test = zeros(Float64, I)
        T0_test = zeros(Float64, I)

        Ti_pred = zeros(Float64, I)
        A0_pred = zeros(Float64, I)
        R0_pred = zeros(Float64, I)
        T0_pred = zeros(Float64, I)

        for i = 1:I
            Ti_pred[i], A0_pred[i], R0_pred[i], T0_pred[i] = RTO(Ai_test[i], Ri_ss)
        end

        for i = 1:I
            Ti_test[i], A0_test[i], R0_test[i], T0_test[i] =
                SurrogateRTO(Ai_test[i], Ri_ss, P, Qc)
        end

        Result2[ro, colum] = norm(Ti_pred - Ti_test, 1)
        println(Result2[ro, colum])
        colum += 1
    end
    # df1 = DataFrame(index = 1:1:100, Sparse = Err_evol_1[1], Cubic = Err_evol_1[2], Quadratic = Err_evol_1[3], Linear = Err_evol_1[4], Constant = Err_evol_1[5])
end
temp_df = DataFrame(Result2, :auto)
CSV.write("full_pred_error_type_1.csv", temp_df)

# df2 = DataFrame(index = 1:1:100, Sparse = suboptimality[1], Cubic = suboptimality[2], Quadratic = suboptimality[3], Linear = suboptimality[4], Constant = suboptimality[5])
# df3 = DataFrame(index = 1:1:5, train_loss = objective, inf_error = Err_evol_inf)

# figure()
# PyPlot.rc("font", family="sans")
# PyPlot.rc("font", size = 12)
# fig, ax = PyPlot.subplots()
# bp = ax.boxplot(Result2, 0, "")
# PyPlot.setp(bp["boxes"], color="black")
# PyPlot.setp(bp["whiskers"], color="black")
# PyPlot.setp(bp["fliers"], color="red")
# PyPlot.setp(bp["medians"], color="red")
# # ax.set_ylim(-0.5,4)
# ax.set_xticklabels(["5", "10", "25", "50", "100", "250", "500"], fontsize = 12)
# ax.tick_params(direction = "in")
# gcf()
# PyPlot.savefig("TrainingDataSize_sparse.png", bbox_inches="tight")
# close()
