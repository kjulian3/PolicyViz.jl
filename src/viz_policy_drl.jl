export viz_policy_drl
function viz_policy_drl(neuralNetworkPath::AbstractString, alpha::Matrix{Float64}=[1.0 1.0]; batch_size::Int64=500,XandY::Bool = false)
    
    nnet = NNet(neuralNetworkPath);
    if length(alpha)!=2
        if !XandY
            grid = RectangleGrid(Ranges, Thetas, Bearings, Speeds, Speeds)
        else
            grid = RectangleGrid(Xs, Ys, Bearings, Speeds, Speeds)
        end
        policy = read_policy(Actions', alpha)
    end
    

    
    c = RGB{U8}(1.,1.,1.) # white
    e = RGB{U8}(.0,.0,.5) # pink
    a = RGB{U8}(.0,.600,.0) # green
    d = RGB{U8}(.5,.5,.5) # grey
    b = RGB{U8}(.7,.9,.0) # neon green
    f = RGB{U8}(0.94,1.0,.7) # pale yellow
    colors =[a; b; f; c; d; e]
   
    @manipulate for psi_int  = round(rad2deg(psis_drl)),
        v_own = sos_drl,
        v_int = sis_drl,
        zoom = [4, 3, 2, 1.5,1],
        nbin = [100,150,200,250],
        interp = [true, false]
            
        #mat =  ccall((:load_network,LIB_BLAS),Ptr{Void},(Ptr{UInt8},),neuralNetworkPath)
        inputsNet= zeros(nbin*nbin,STATE_DIM)    
        ind = 1
        for i=linspace(round(Int,-1*RangeMax/zoom),round(Int,RangeMax/zoom),nbin)
            for j=linspace(round(Int,-1*RangeMax/zoom),round(Int,RangeMax/zoom),nbin)  
                r = sqrt(i^2+j^2)
                th = atan2(j,i)
                inputsNet[ind,:] = [r,th,deg2rad(psi_int),v_own,v_int];
                ind = ind+1
            end
        end            

        q_nnet = zeros(nbin*nbin,ACTION_DIM);
        ind = 1
        while ind+batch_size<nbin*nbin            
            input = inputsNet[ind:(ind+batch_size-1),:]'
            output = evaluate_network_multiple(nnet,input) 
            q_nnet = [q_nnet[1:(ind-1),:];output';q_nnet[ind+batch_size:end,:]]
            ind=ind+batch_size
        end
        input = inputsNet[ind:end,:]'
        output = evaluate_network_multiple(nnet,input)
        q_nnet = [q_nnet[1:(ind-1),:];output']
        
        
        
        
        ind = 1 
        function get_heat1(x::Float64, y::Float64) 
            if !XandY
                r = norm([x,y])
                th = atan2(y,x)
                qvals = evaluate(policy, get_belief(
                [r, th, deg2rad(psi_int), v_own, v_int], grid,interp,true)) 
            else
                qvals = evaluate(policy, get_belief(
                [x, y, deg2rad(psi_int), v_own, v_int], grid,interp,true,true)) 
            end
            return Actions[indmax(qvals)]
        end # function get_heat1
        
        function get_heat2(x::Float64, y::Float64)              
            r = sqrt(x^2+y^2)
            th = atan2(y,x)            
            action  = Actions[indmax(q_nnet[ind,:])]
            ind = ind+1
            return action
        end # function get_heat2
        
        g = GroupPlot(2, 1, groupStyle = "horizontal sep=3cm")
        if length(alpha)!=2
            push!(g, Axis([
                Plots.Image(get_heat1, (round(Int,-1*RangeMax/zoom), round(Int,RangeMax/zoom)), 
                (round(Int,-1*RangeMax/zoom), round(Int,RangeMax/zoom)), 
                           zmin = -20, zmax = 20,
                           xbins = nbin, ybins = nbin,
                           colormap = ColorMaps.RGBArray(colors), colorbar = false),
               Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
               Plots.Node(L">", 2500/zoom, 2500/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
                ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Discrete Value Iteration Policy"))
        end
        ind = 1
        push!(g, Axis([
            Plots.Image(get_heat2, (round(Int,-1*RangeMax/zoom), round(Int,RangeMax/zoom)), 
            (round(Int,-1*RangeMax/zoom), round(Int,RangeMax/zoom)), 
                       zmin = -20, zmax = 20,
                       xbins = nbin, ybins = nbin,
                       colormap = ColorMaps.RGBArray(colors)),
           Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
           Plots.Node(L">", 2500/zoom, 2500/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
            ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Neural Net Policy"))
        g
    end # for p_int, v0, v1, pa, ta
end # function viz_policy_drl