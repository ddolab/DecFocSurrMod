function penaltyIOP(S, inlet_A, inlet_R, type)
    k = 1

    c1 = 500 * ones(Float64, S, 4) # 500
    c2 = 500 * ones(Float64, S, 3)
    # c3 = 500*ones(Float64, NoS, 3, T)
    eta = 1000 # 1000

    A = initialize(S, inlet_A, inlet_R)
    Ahat = deepcopy(A0_train)
    Rhat = deepcopy(R0_train)
    That = deepcopy(T0_train)
    Tihat = deepcopy(Ti_train)

    max_iter = 70
    global A, Qc, c1, c2, lambda, P, Ahat, Rhat, That, Tihat, obj
    while k <= max_iter
        println("Iteration:", k)
        # global A, B, C, Qc, R, mu, c1, c2, c3, xhat, uhat
        Qc, lambda = Step2(S, inlet_A, inlet_R, A, Ahat, Rhat, That, Tihat, c1, c2)
        if flag == 1
            break
        end
        A, P, d1, d2, obj =
            Step3(S, inlet_A, inlet_R, Ahat, Rhat, That, Tihat, lambda, Qc, c1, c2, type)
        if flag == 1
            break
        end
        Ahat, Rhat, That, Tihat = Step1(S, inlet_A, inlet_R, A, Qc, lambda, c1, c2)
        if flag == 1
            break
        end
        println(obj)
        println(norm(d1))
        println(norm(d2))
        c1 += eta .* d1
        c2 += eta .* d2
        k += 1
    end
    return Qc, P, obj
end
