using JuMP, LinearAlgebra, Ipopt, Gurobi, Random, JLD2
include("fop.jl")
include("surrogate_fop.jl")
include("iop.jl")

rng = MersenneTwister(12345)
samples = [5, 10, 20, 50, 75, 100, 200]
n_instances = 10

p = 4
l = 1
r = 4
nj = [2, 1, 2, 1]

c = [6 10 3 7; 16 0 13 0] / 1000
d = [9, 15, 6, 12]
f_bar = [0 0 50 350; 200 0 0 0]

lambda = zeros(Float64, maximum(nj), 4, l)
lambda[1, 1, 1] = 3
lambda[2, 1, 1] = 1
lambda[1, 2, 1] = 2
lambda[1, 3, 1] = 3.5
lambda[2, 3, 1] = 1.5
lambda[1, 4, 1] = 2.5

z = zeros(Float64, r, l)
z[1, 1] = 2.5
z[2, 1] = 1.5
z[3, 1] = 3
z[4, 1] = 2

n_bilinear = p * r * l
bilinear_map = zeros(Int, p, r, l)
count = 1
for j = 1:p
    for k = 1:r
        for w = 1:l
            bilinear_map[j, k, w] = count
            global count += 1
        end
    end
end
n_degree = 3 # order of polynomial
n_input = 4
pred_error = zeros(Float64, n_instances, length(samples))

for instance = 1:n_instances
    global n_samples = maximum(samples)
    global demand = zeros(Float64, n_samples, r)
    u = rand(rng, 100:200, n_samples, n_input) / 100
    for i = 1:n_samples
        demand[i, :] .= u[i, :] * 100
    end

    ff = zeros(Float64, n_samples, maximum(nj), p)
    xx = zeros(Float64, n_samples, p, r)
    qq = zeros(Float64, n_samples, p, l)
    for s = 1:n_samples
        temp, ff[s, :, :], xx[s, :, :], qq[s, :, :] = fop(demand[s, :])
    end

    n_test = 100
    test_demand = zeros(Float64, n_test, r)
    test_u = rand(rng, 100:200, n_test, n_input) / 100
    for i = 1:n_test
        test_demand[i, :] .= test_u[i, :] * 100
    end
    ff_true = zeros(Float64, n_test, maximum(nj), p)
    xx_true = zeros(Float64, n_test, p, r)
    obj_true = zeros(Float64, n_test)
    for i = 1:n_test
        obj_true[i], ff_true[i, :, :], xx_true[i, :, :], temp = fop(test_demand[i, :])
    end

    for column = 1:length(samples)
        global n_samples = samples[column]

        params = iop(
            u[1:n_samples, :],
            ff[1:n_samples, :, :],
            xx[1:n_samples, :, :],
            qq[1:n_samples, :, :],
        )

        ff_hat = zeros(Float64, n_test, maximum(nj), p)
        xx_hat = zeros(Float64, n_test, p, r)
        obj_hat = zeros(Float64, n_test)
        for i = 1:n_test
            obj_hat[i], ff_hat[i, :, :], xx_hat[i, :, :] =
                surrogate_fop(test_u[i, :], test_demand[i, :], params)
        end

        pred_error[instance, column] = norm(obj_true - obj_hat, 1)
        jldsave("surrogate_parameters_$(instance)_$(samples[column]).jld2"; params)
    end
end
jldsave("prediction_error.jld2"; pred_error)
