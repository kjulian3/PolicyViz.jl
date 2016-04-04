export get_belief,get_qval!,Policy,read_policy,evaluate

type Policy
    alpha       :: Matrix{Float64}
    actions     :: Matrix{Float64}
    nactions    :: Int64
    qvals       :: Vector{Float64}

    function Policy(alpha::Matrix{Float64}, actions::Matrix{Float64})
        return new(alpha, actions, size(actions, 2), zeros(size(actions, 2)))
    end # function Policy
end

function read_policy(actions::Matrix{Float64}, alpha::Matrix{Float64})
    return Policy(alpha, actions)
end # function read_policy

function evaluate(policy::Policy, belief::SparseMatrixCSC{Float64,Int64})
    fill!(policy.qvals, 0.0)
    get_qval!(policy, belief)
    return copy(policy.qvals)
end # function evaluate

function get_qval!(policy::Policy, belief::SparseMatrixCSC{Float64, Int64})
    fill!(policy.qvals, 0.0)
    for iaction in 1:policy.nactions
        for ib in 1:length(belief.rowval)
            policy.qvals[iaction] += belief.nzval[ib] * policy.alpha[belief.rowval[ib], iaction]
        end # for b
    end # for iaction
    #println(policy.qvals)
end # function get_qval!

function get_belief(pstate::Vector{Float64}, grid::RectangleGrid,interp::Bool=false,drl::Bool=false,XandY::Bool=false)
    belief = spzeros(NSTATES, 1)
    if drl
        if XandY
            belief = spzeros(NSTATES_drl_xandy,1)
        else
            belief = spzeros(NSTATES_drl,1)
        end
    end
    indices, weights = interpolants(grid, pstate)
    if !interp
        largestWeight = 0;
        largestIndex = 0;
        for i = 1:length(weights)
            if weights[i]>largestWeight
                largestWeight = weights[i]
                largestIndex = indices[i]
            end
        end
        indices = largestIndex
        weights = 1.0
    end
    for i = 1:length(indices)
        belief[indices[i]] = weights[i]
    end # for i
    return belief
end # function get_belief


function belief_states(r,th,psi_int,v_own,v_int,tau,pa,deltaR, deltaPsi,deltaV, deltaTheta,nnet)
    belief = zeros(9,7)
    pastActions = pas
    if nnet
        pastActions = pasTrue
    end
    belief[1,:] = [r,th,deg2rad(psi_int),v_own,v_int,tau,pastActions[pa]];
    belief[2,:] = [r+deltaR,th,deg2rad(psi_int),v_own,v_int,tau,pastActions[pa]];

    rTemp = r-deltaR
    if rTemp < 0
        rTemp = 0
    end
    belief[3,:] = [rTemp,th,deg2rad(psi_int),v_own,v_int,tau,pastActions[pa]];

    psiTemp = psi_int+deltaPsi
    if psiTemp>180.0
        psiTemp-=360.0
    end
    belief[4,:] = [r,th,deg2rad(psiTemp),v_own,v_int,tau,pastActions[pa]];

    psiTemp = psi_int-deltaPsi
    if psiTemp<-180.0
        psiTemp+=360.0
    end
    belief[5,:] = [r,th,deg2rad(psiTemp),v_own,v_int,tau,pastActions[pa]];
    belief[6,:] = [r,th,deg2rad(psi_int),v_own,v_int+deltaV,tau,pastActions[pa]];

    vTemp = v_int-deltaV
    if vTemp < 0
        vTemp = 0
    end
    belief[7,:] = [r,th,deg2rad(psi_int),v_own,vTemp,tau,pastActions[pa]];

    thTemp = th+deltaTheta
    if thTemp>pi
        thTemp-=2*pi
    end
    belief[8,:] = [r,thTemp,deg2rad(psi_int),v_own,v_int,tau,pastActions[pa]];

    thTemp = th-deltaTheta
    if thTemp<-pi
        thTemp+=2*pi
    end
    belief[9,:] = [r,thTemp,deg2rad(psi_int),v_own,v_int,tau,pastActions[pa]];
    
    return belief
    
end #function belief_states