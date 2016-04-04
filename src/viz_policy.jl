export viz_policy

function viz_policy(alpha::Matrix{Float64}, neuralNetworkPath::AbstractString, batch_size::Int64=500)
    
    nnet = NNet(neuralNetworkPath);
    grid  = RectangleGrid(pas,taus,sis,sos,psis,thetas,ranges) 
    grid2 = RectangleGrid(thetas,ranges)
    
    policy = read_policy(ACTIONS, alpha)
    c = RGB{U8}(1.,1.,1.) # white
    e = RGB{U8}(.0,.0,.5) # pink
    a = RGB{U8}(.0,.600,.0) # green
    d = RGB{U8}(.5,.5,.5) # grey
    b = RGB{U8}(.7,.9,.0) # neon green
    colors = [a; b; c; d; e]
    
    
    @manipulate for psi_int  = convert(Array{Int32,1},round(rad2deg(psis))),
        v_own = sos,
        v_int = sis,
        tau in taus,
        pa = pas,
        zoom = [4, 3, 2, 1],
        nbin = [100,150,200,250],
        Interp = [false,true],
        Belief = [false,true],
        beliefProb = [0.333,0.111,0.01],
        
        deltaR   = [40, 400, 4000, 0],
        deltaTh = [5.0, 30, 60, 0],
        deltaPsi = [5, 30, 60,90, 0],
        deltaV   = [10, 100, 200, 0],
        worst    = [false,true]
        
            
            
        deltaTheta = deltaTh*pi/180.0
        #Load table with the inputs needed to plot the heat map
        if Belief
            numBelief = 9
            inputsNet= zeros(nbin*nbin*numBelief,7)    
            ind = 1
            for i=linspace(round(Int,-1*RANGEMAX/zoom),round(Int,RANGEMAX/zoom),nbin)
                for j=linspace(round(Int,-1*RANGEMAX/zoom),round(Int,RANGEMAX/zoom),nbin)
                    r = sqrt(i^2+j^2)
                    th = atan2(j,i)
                    bel = belief_states(r,th,psi_int,v_own,v_int,tau,pa,deltaR,deltaPsi,deltaV,deltaTheta,true)
                    inputsNet[ind:ind+8,:] = bel
                    ind = ind+numBelief
                end
            end
        else
            numBelief = 1
            inputsNet= zeros(nbin*nbin,7)    
            ind = 1
            for i=linspace(round(Int,-1*RANGEMAX/zoom),round(Int,RANGEMAX/zoom),nbin)
                for j=linspace(round(Int,-1*RANGEMAX/zoom),round(Int,RANGEMAX/zoom),nbin)
                    r = sqrt(i^2+j^2)
                    th = atan2(j,i)
                    inputsNet[ind,:] = [r,th,deg2rad(psi_int),v_own,v_int,tau,pasTrue[pa]];
                    ind = ind+1
                end
            end
        end
        
        #Calculate all of the Q values from the input array
        q_nnet = zeros(nbin*nbin*numBelief,5);
        ind = 1
        
        while ind+batch_size<nbin*nbin*numBelief
            input = inputsNet[ind:(ind+batch_size-1),:]'
            output = evaluate_network_multiple(nnet,input) 
            q_nnet = [q_nnet[1:(ind-1),:];output';q_nnet[ind+batch_size:end,:]]
            ind=ind+batch_size
        end
        input = inputsNet[ind:end,:]'
        output = evaluate_network_multiple(nnet,input)
        q_nnet = [q_nnet[1:(ind-1),:];output']

        
        ind = 1
        # Q Table Heat Map
        function get_heat1(x::Float64, y::Float64)
            r = sqrt(x^2+y^2)
            th = atan2(y,x)
            bel = belief_states(r,th,psi_int,v_own,v_int,tau,pa,deltaR,deltaPsi,deltaV,deltaTheta,false)
            qvals = evaluate(policy, get_belief(bel[1,end:-1:1][:],grid,Interp))
            if Belief
                if !worst
                    qvals*=beliefProb
                end
                for i=2:9
                    temp = evaluate(policy, get_belief(bel[i,end:-1:1][:],grid,Interp))
                    if worst
                        if minimum(temp)>minimum(qvals)
                            qvals = temp
                        end
                    else
                        qvals += temp*(1.0-beliefProb)/(numBelief-1.0)

                    end
                end
            end
            return rad2deg(ACTIONS[indmin(qvals)])
       end # function get_heat1
        
        
        #Neural Net Heat Map
       function get_heat2(x::Float64, y::Float64)              
           r = sqrt(x^2+y^2)
           th = atan2(y,x)            
           qvals = q_nnet[ind,:]
            if !worst
                qvals*=beliefProb
            end
           if Belief
               for i = 1:8
                    qvalTemp = q_nnet[ind+i,:]
                    if worst
                        if minimum(qvalTemp)>minimum(qvals)
                            qvals = qvalTemp
                        end
                    else
                        qvals+=qvalTemp*(1.0-beliefProb)/(numBelief-1.0)
                    end
               end
           end

           ind +=numBelief
           return rad2deg(ACTIONS[indmin(qvals)])
       end # function get_heat2
        
        if Belief
            g = GroupPlot(2, 2, groupStyle = "horizontal sep=3cm, vertical sep = 3cm")
            Belief = false
            push!(g, Axis([
                Plots.Image(get_heat1, (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           zmin = -3, zmax = 3,
                           xbins = nbin, ybins = nbin,
                colormap = ColorMaps.RGBArray(colors), colorbar=false),
               Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
               Plots.Node(L">", 53000/zoom, 53000/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
                    ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Nominal Q Table action"))
        
            
           push!(g, Axis([
               Plots.Image(get_heat2, (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           zmin = -3, zmax = 3,
                           xbins = nbin, ybins = nbin,
                           colormap = ColorMaps.RGBArray(colors)),
               Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
               Plots.Node(L">", 53000/zoom, 53000/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
                    ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Nominal Neural Net action"))
            
            
            Belief = true
            ind = 1
            push!(g, Axis([
                Plots.Image(get_heat1, (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           zmin = -3, zmax = 3,
                           xbins = nbin, ybins = nbin,
                colormap = ColorMaps.RGBArray(colors), colorbar=false),
               Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
               Plots.Node(L">", 53000/zoom, 53000/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
                    ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Belief Q Table action"))

            push!(g, Axis([
               Plots.Image(get_heat2, (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           zmin = -3, zmax = 3,
                           xbins = nbin, ybins = nbin,
                           colormap = ColorMaps.RGBArray(colors)),
               Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
               Plots.Node(L">", 53000/zoom, 53000/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
                    ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Belief Neural Net action"))
        
        else
            g = GroupPlot(2, 1, groupStyle = "horizontal sep=3cm")
            push!(g, Axis([
                Plots.Image(get_heat1, (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           zmin = -3, zmax = 3,
                           xbins = nbin, ybins = nbin,
                colormap = ColorMaps.RGBArray(colors), colorbar=false),
               Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
               Plots.Node(L">", 53000/zoom, 53000/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
                ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Q Table action"))
            
            push!(g, Axis([
               Plots.Image(get_heat2, (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           zmin = -3, zmax = 3,
                           xbins = nbin, ybins = nbin,
                           colormap = ColorMaps.RGBArray(colors)),
               Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
               Plots.Node(L">", 53000/zoom, 53000/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
                ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Neural Net action"))
        end
        g
    end # for p_int, v0, v1, pa, ta
end # function viz_pairwise_policy


function viz_policy(neuralNetworkPath::AbstractString, batch_size=500)
    
    nnet = NNet(neuralNetworkPath);
       
    c = RGB{U8}(1.,1.,1.) # white
    e = RGB{U8}(.0,.0,.5) # pink
    a = RGB{U8}(.0,.600,.0) # green
    d = RGB{U8}(.5,.5,.5) # grey
    b = RGB{U8}(.7,.9,.0) # neon green
    colors =[a; b; c; d; e]
    
    
    @manipulate for psi_int  = convert(Array{Int32,1},round(rad2deg(psis))),
        v_own = sos,
        v_int = sis,
        tau in taus,
        pa = pas,
        zoom = [4, 3, 2, 1],
        nbin = [100,150,200,250],
        Belief = [false,true],
        beliefProb = [0.333,0.111,0.01],
        
        deltaR   = [40, 400, 4000, 0],
        deltaTh  = [5.0, 30, 60, 0],
        deltaPsi = [5, 30, 60,90, 0],
        deltaV   = [10, 100, 200, 0],
        worst    = [false,true]
        
            
            
        deltaTheta = deltaTh*pi/180.0
        
        #Load table with the inputs needed to plot the heat map
        if Belief
            numBelief = 9
            inputsNet= zeros(nbin*nbin*numBelief,7)    
            ind = 1
            for i=linspace(round(Int,-1*RANGEMAX/zoom),round(Int,RANGEMAX/zoom),nbin)
                for j=linspace(round(Int,-1*RANGEMAX/zoom),round(Int,RANGEMAX/zoom),nbin)
                    r = sqrt(i^2+j^2)
                    th = atan2(j,i)
                    bel = belief_states(r,th,psi_int,v_own,v_int,tau,pa,deltaR,deltaPsi,deltaV,deltaTheta,true)
                    inputsNet[ind:ind+8,:] = bel
                    ind = ind+numBelief
                end
            end
        else
            numBelief = 1
            inputsNet= zeros(nbin*nbin,7)    
            ind = 1
            for i=linspace(round(Int,-1*RANGEMAX/zoom),round(Int,RANGEMAX/zoom),nbin)
                for j=linspace(round(Int,-1*RANGEMAX/zoom),round(Int,RANGEMAX/zoom),nbin)
                    r = sqrt(i^2+j^2)
                    th = atan2(j,i)
                    inputsNet[ind,:] = [r,th,deg2rad(psi_int),v_own,v_int,tau,pasTrue[pa]];
                    ind = ind+1
                end
            end
        end
        
        #Calculate all of the Q values from the input array
        q_nnet = zeros(nbin*nbin*numBelief,5);
        ind = 1
        
        while ind+batch_size<nbin*nbin*numBelief
            input = inputsNet[ind:(ind+batch_size-1),:]'
            output = evaluate_network_multiple(nnet,input) 
            q_nnet = [q_nnet[1:(ind-1),:];output';q_nnet[ind+batch_size:end,:]]
            ind=ind+batch_size
        end
        input = inputsNet[ind:end,:]'
        output = evaluate_network_multiple(nnet,input)
        q_nnet = [q_nnet[1:(ind-1),:];output']

        
        ind = 1       
        #Neural Net Heat Map
       function get_heat2(x::Float64, y::Float64)              
           r = sqrt(x^2+y^2)
           th = atan2(y,x)            
           qvals = q_nnet[ind,:]
            if !worst
                qvals*=beliefProb
            end
           if Belief
               for i = 1:8
                    qvalTemp = q_nnet[ind+i,:]
                    if worst
                        if minimum(qvalTemp)>minimum(qvals)
                            qvals = qvalTemp
                        end
                    else
                        qvals+=qvalTemp*(1.0-beliefProb)/(numBelief-1.0)
                    end
               end
           end

           ind +=numBelief
           return rad2deg(ACTIONS[indmin(qvals)])
       end # function get_heat2
        
        if Belief
            g = GroupPlot(2, 1, groupStyle = "horizontal sep=3cm, vertical sep = 3cm")
            Belief = false
            
           push!(g, Axis([
               Plots.Image(get_heat2, (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           zmin = -3, zmax = 3,
                           xbins = nbin, ybins = nbin,
                           colormap = ColorMaps.RGBArray(colors)),
               Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
               Plots.Node(L">", 53000/zoom, 53000/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
                    ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Nominal Neural Net action"))
            
            
            Belief = true
            ind = 1
            push!(g, Axis([
               Plots.Image(get_heat2, (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           zmin = -3, zmax = 3,
                           xbins = nbin, ybins = nbin,
                           colormap = ColorMaps.RGBArray(colors)),
               Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
               Plots.Node(L">", 53000/zoom, 53000/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
                    ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Belief Neural Net action"))
        
        else
            g = GroupPlot(1, 1, groupStyle = "horizontal sep=3cm")
            push!(g, Axis([
               Plots.Image(get_heat2, (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           zmin = -3, zmax = 3,
                           xbins = nbin, ybins = nbin,
                           colormap = ColorMaps.RGBArray(colors)),
               Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
               Plots.Node(L">", 53000/zoom, 53000/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
                ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Neural Net action"))
        end
        g
    end # for p_int, v0, v1, pa, ta, etc
end # function viz_policy