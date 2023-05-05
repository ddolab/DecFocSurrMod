function fop(s, solver)
    t_start = time()
    if solver == "global"
        fop = Model(() -> Gurobi.Optimizer(GRB_ENV))
        set_optimizer_attribute(fop, "OutputFlag", 0)
        set_optimizer_attribute(fop, "NonConvex", 2)
        set_optimizer_attribute(fop, "TimeLimit", 100)
    else
        fop = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    end

<<<<<<< HEAD
    JuMP.@variables(
        fop,
        begin
            f[i = 1:maximum(nj), j = 1:p] >= 0, (start = rand(rng, maximum(nj), p)[i, j])
            q[1:p, 1:l] >= 0
            x[1:p, 1:r] >= 0
        end
    )

    @constraints(
        fop,
        begin
            [j = 1:p], sum(f[i, j] for i = 1:nj[j]) - sum(x[j, k] for k = 1:r) == 0
            [j = 1:p, w = 1:l],
            q[j, w] * sum(x[j, k] for k = 1:r) -
            sum(lambda[i, j, w] * f[i, j] for i = 1:nj[j]) == 0
            [k = 1:r], sum(x[j, k] for j = 1:p) <= s[k]
            [k = 1:r, w = 1:l],
            sum((q[j, w]) * x[j, k] for j = 1:p) - z[k, w] * sum(x[j, k] for j = 1:p) <= 0
        end
    )

    @objective(
        fop,
        Min,
        sum(1000 * c[i, j] * (f[i, j]) for j = 1:p, i = 1:nj[j]) +
        sum(c[i, j] * (f[i, j] - f_bar[i, j])^2 for j = 1:p, i = 1:nj[j]) -
        sum(d[k] * x[j, k] for j = 1:p, k = 1:r)
    )
    optimize!(fop)
    return time() - t_start, objective_value(fop), value.(f), value.(x)
=======
    @JuMP.variables(fop, begin
        f[i = 1:maximum(nj), j = 1:p] >= 0, (start = rand(rng, maximum(nj), p)[i, j])
        q[1:p, 1:l] >= 0
        x[1:p, 1:r] >= 0
    end)

    @constraints(fop, begin
        [j = 1:p], sum(f[i, j] for i in 1:nj[j]) - sum(x[j, k] for k in 1:r) == 0
        [j = 1:p, w = 1:l], q[j, w]*sum(x[j, k] for k in 1:r) - sum(lambda[i, j, w]*f[i, j] for i in 1:nj[j]) == 0
        [k = 1:r], sum(x[j, k] for j in 1:p) <= s[k]
        [k = 1:r, w = 1:l], sum((q[j, w])*x[j, k] for j in 1:p) - z[k, w]*sum(x[j, k] for j in 1:p) <= 0
    end)

    @objective(fop, Min, sum(1000*c[i, j]*(f[i, j]) for j in 1:p, i in 1:nj[j]) + 
    sum(c[i, j]*(f[i, j] - f_bar[i, j])^2 for j in 1:p, i in 1:nj[j]) - 
    sum(d[k]*x[j, k] for j in 1:p, k in 1:r))
    optimize!(fop)
    return time()-t_start, objective_value(fop), value.(f), value.(x)
>>>>>>> 7b1153bdf22863c45f588b2f9307f2a426275741
end
