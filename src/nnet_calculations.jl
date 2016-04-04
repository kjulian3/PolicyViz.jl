#Julia implementation of "load_network" function
type NNet
    file::AbstractString
    weights::Array{Any,1}
    biases::Array{Any,1}
    symmetric::Int32
    numLayers::Int32
    inputSize::Int32
    outputSize::Int32
    maxLayerSize::Int32
    
    layerSizes::Array{Int32,1}
    mins::Array{Float64,1}
    maxes::Array{Float64,1}
    means::Array{Float64,1}
    ranges::Array{Float64,1}
    
    function NNet(file::AbstractString)
        this  = new()
        this.file = file
        f = open(this.file)
        line = readline(f)
        line = readline(f)
        record = split(line,[',','\n'])
        this.numLayers = parse(Int32,record[1])
        this.inputSize = parse(Int32,record[2])
        this.outputSize = parse(Int32,record[3])
        this.maxLayerSize=parse(Int32,record[4])
        
        line = readline(f)
        record = split(line,[',','\n'])
        this.layerSizes = zeros(this.numLayers+1)
        for i=1:(this.numLayers+1)
            this.layerSizes[i]=parse(Int32,record[i])
        end
        
        line = readline(f)
        record = split(line,[',','\n'])
        this.symmetric = parse(Int32,record[1])
        
        line = readline(f)
        record = split(line,[',','\n'])
        this.mins = zeros(this.inputSize)
        for i=1:(this.inputSize)
            this.mins[i]=parse(Float64,record[i])
        end
        
        line = readline(f)
        record = split(line,[',','\n'])
        this.maxes = zeros(this.inputSize)
        for i=1:(this.inputSize)
            this.maxes[i]=parse(Float64,record[i])
        end
        
        
        line = readline(f)
        record = split(line,[',','\n'])
        this.means = zeros(this.inputSize+1)
        for i=1:(this.inputSize+1)
            this.means[i]=parse(Float64,record[i])
        end
        
        line = readline(f)
        record = split(line,[',','\n'])
        this.ranges = zeros(this.inputSize+1)
        for i=1:(this.inputSize+1)
            this.ranges[i]=parse(Float64,record[i])
        end
        
        
        this.weights = Any[zeros(this.layerSizes[2],this.layerSizes[1])]
        this.biases  = Any[zeros(this.layerSizes[2])]
        for i=2:this.numLayers
            this.weights = [this.weights;Any[zeros(this.layerSizes[i+1],this.layerSizes[i])]]
            this.biases  = [this.biases;Any[zeros(this.layerSizes[i+1])]]
        end
        
        layer=1
        i=1
        j=1
        line = readline(f)
        record = split(line,[',','\n'])
        while !eof(f)
            while i<=this.layerSizes[layer+1]
                while record[j]!=""
                    this.weights[layer][i,j] = parse(Float64,record[j])
                    j=j+1
                end
                j=1
                i=i+1
                line = readline(f)
                record = split(line,[',','\n'])
            end
            i=1
            while i<=this.layerSizes[layer+1]
                this.biases[layer][i] = parse(Float64,record[1])
                i=i+1
                line = readline(f)
                record = split(line,[',','\n'])
            end
            layer=layer+1
            i=1
            j=1
        end
        close(f)
        
        return this
    end
end

#Evaluates one set of inputs
function evaluate_network(nnet::NNet,input::Array{Float64,1})
    numLayers = nnet.numLayers
    inputSize = nnet.inputSize
    outputSize = nnet.outputSize
    symmetric = nnet.symmetric
    biases = nnet.biases
    weights = nnet.weights
    
    inputs = zeros(inputSize)
    for i = 1:inputSize
        if input[i]<nnet.mins[i]
            inputs[i] = (nnet.mins[i]-nnet.means[i])/nnet.ranges[i]
        elseif input[i] > nnet.maxes[i]
            inputs[i] = (nnet.maxes[i]-nnet.means[i])/nnet.ranges[i] 
        else
            inputs[i] = (input[i]-nnet.means[i])/nnet.ranges[i] 
        end
    end
    if symmetric ==1 && inputs[2]<0
        inputs[2] = -inputs[2]
        inputs[1] = -inputs[1]
    else
        symmetric = 0
    end
    for layer = 1:numLayers-1
        temp = max(*(weights[layer],inputs[1:nnet.layerSizes[layer]])+biases[layer],0)
        inputs = temp
    end
    outputs = *(weights[end],inputs[1:nnet.layerSizes[end-1]])+biases[end]
    for i=1:outputSize
        outputs[i] = outputs[i]*nnet.ranges[end]+nnet.means[end]
    end
    return outputs
end

#Evaluates multiple inputs at once. Each set of inputs should be a column in the input array
#Returns a column of output Q values for each input set
function evaluate_network_multiple(nnet::NNet,input::Array{Float64,2})
    numLayers = nnet.numLayers
    inputSize = nnet.inputSize
    outputSize = nnet.outputSize
    symmetric = nnet.symmetric
    biases = nnet.biases
    weights = nnet.weights
        
    _,numInputs = size(input)
    symmetryVec = zeros(numInputs)
    
    inputs = zeros(inputSize,numInputs)
    for i = 1:inputSize
        for j = 1:numInputs
            if input[i,j]<nnet.mins[i]
                inputs[i,j] = (nnet.mins[i]-nnet.means[i])/nnet.ranges[i]
            elseif input[i,j] > nnet.maxes[i]
                inputs[i,j] = (nnet.maxes[i]-nnet.means[i])/nnet.ranges[i] 
            else
                inputs[i,j] = (input[i,j]-nnet.means[i])/nnet.ranges[i] 
            end
            
            
            inputs[i,j] = (input[i,j]-nnet.means[i])/nnet.ranges[i] 
        end
    end
    for i=1:numInputs
        if symmetric ==1 && inputs[2,i]<0
            inputs[2,i] = -inputs[2,i]
            inputs[1,i] = -inputs[1,i]
            symmetryVec[i] = 1
        else
            symmetryVec[i] = 0
        end
    end
    
    for layer = 1:numLayers-1
        inputs = max(*(weights[layer],inputs[1:nnet.layerSizes[layer],:])+*(biases[layer],ones(1,numInputs)),0)
    end
    outputs = *(weights[end],inputs[1:nnet.layerSizes[end-1],:])+*(biases[end],ones(1,numInputs))
    for i=1:outputSize
        for j=1:numInputs
            outputs[i,j] = outputs[i,j]*nnet.ranges[end]+nnet.means[end]
        end
    end
    return outputs
end