module LossFunctions
#-----------------------------------------------------------
#PURPOSE
#	Colin T. Bowers module for loss functions
#NOTES
#	To add a new loss function, provide a new type definition and any constructors under the TYPE DEFINTIIONS heading, then add an appropriate loss method for scalar inputs.
#	If you want to aggregate your loss series in some way, you should probably look at using the Distances package before thinking about this package
#LICENSE
#	MIT License (see github repository for more detail: https://github.com/colintbowers/LossFunctions.jl.git)
#-----------------------------------------------------------

#Load any entire modules that are needed (use import ModuleName1, ModuleName2, etc)

#Load any specific variables/functions that are needed (use import ModuleName1.FunctionName1, ModuleName2.FunctionName2, etc)
import 	Base.string,
		Base.show

#Specify the variables/functions to export (use export FunctionName1, FunctionName2, etc)
export 	LossFunction, #Abstract supertype
		SquaredLoss, #Loss function type
		AbsoluteLoss, #Loss function type
		MinkowskiLoss, #Loss function type
		SquaredLogLoss, #Loss function type
		AbsoluteLogLoss, #Loss function type
		SquaredPropLoss, #Loss function type
		AbsolutePropLoss, #Loss function type
		QLIKE, #Loss function type
		QLIKEReverse, #Loss function type
		loss, #Generic method for calculating loss between two things
		lossdiff #Generic method for calculating a loss differential, ie L(x-BaseCase) - L(y-BaseCase)


#******************************************************************************

#----------------------------------------------------------
#SET CONSTANTS FOR MODULE
#----------------------------------------------------------
#Set constants here




#----------------------------------------------------------
#LOSS FUNCTION TYPES
#FIELDS
#	Fields are unique to each type and hold the parameters necessary to implement that particular loss function
#PURPOSE
#	Each type stores the information needed to evaluate a loss function
#CONSTRUCTORS
#	Outer constructors are unique to the type and are generally used to provide popular default values.
#METHODS
#	loss(x, y, lF::LossFunction): Evaluate the loss between x and y using the specified loss function
#	string
#	show
#NOTES
#----------------------------------------------------------
#SUPERTYPE
abstract LossFunction
#---- TYPE DEFINTIIONS ----
type DummyLoss <: LossFunction; end #dummy type that does nothing
type SquaredLoss <: LossFunction; end
type AbsoluteLoss <: LossFunction; end
type MinkowskiLoss <: LossFunction
	p::Float64
	MinkowskiLoss(p::Float64) = (p <= 0) ? error("Minkowski parameter must strictly positive") : new(p)
end
MinkowskiLoss() = MinkowskiLoss(2.0) #Default is SquaredLoss
MinkowskiLoss(p::Number) = MinkowskiLoss(convert(Float64, p))
type SquaredLogLoss <: LossFunction; end
type AbsoluteLogLoss <: LossFunction; end
type SquaredPropLoss <: LossFunction; end
type AbsolutePropLoss <: LossFunction; end
type QLIKE <: LossFunction; end
type QLIKEReverse <: LossFunction; end #QLIKE is asymmetric, so it is useful to define a type which swaps the input order
type HuberLoss <: LossFunction
	p::Float64
	HuberLoss(p::Float64) = (p <= 0) ? error("Huber loss parameter must be strictly positive") : new(p)
end
HuberLoss() = HuberLoss(1.0)
HuberLoss(p::Number) = HuberLoss(convert(Float64, p))
#---- string METHODS -------------
string(lF::SquaredLoss) = "squaredLoss"
string(lF::AbsoluteLoss) = "absoluteLoss"
string(lF::MinkowskiLoss) = "minkowskiLoss"
string(lF::SquaredLogLoss) = "squaredLogLoss"
string(lF::AbsoluteLogLoss) = "absoluteLogLoss"
string(lF::SquaredPropLoss) = "squaredProportionalLoss"
string(lF::AbsolutePropLoss) = "absoluteProportionalLoss"
string(lF::QLIKE) = "QLIKE"
string(lF::QLIKEReverse) = "QLIKEReverse"
string(lF::HuberLoss) = "huberLoss"
#---- show METHODS -------------
show{T<:Union(SquaredLoss, AbsoluteLoss, SquaredLogLoss, AbsoluteLogLoss, SquaredPropLoss, AbsolutePropLoss, QLIKE)}(io::IO, lF::T) = println(io, "loss function = " * string(lF))
function show(io::IO, lf::MinkowskiLoss)
	println(io, "loss function = " * string(lF))
	println(io, "    p = " * string(lF.p))
end
show(lf::LossFunction) = show(STDOUT, lf)
#----- tolossfunction METHOD -------
#WARNING: This function is not type stable
function tolossfunction(s::Symbol)
	if s == :squaredLoss ; return(SquaredLoss())
	elseif s == :absoluteLoss ; return(AbsoluteLoss())
	elseif s == :minkowskiLoss ; return(MinkowskiLoss())
	elseif s == :squaredLogLoss ; return(SquaredLogLoss())
	elseif s == :absoluteLogLoss ; return(AbsoluteLogLoss())
	elseif s == :squaredPropLoss ; return(SquaredPropLoss())
	elseif s == :absolutePropLoss ; return(AbsolutePropLoss())
	elseif s == :QLIKE ; return(QLIKE())
	elseif s == :QLIKEReverse ; return(QLIKEReverse())
	elseif s == :huberLoss ; return(HuberLoss())
	else ; error("Invalid input symbol")
	end
end
#---- loss METHODS -------------
loss(x::Number, y::Number, lF::SquaredLoss) = (x - y)^2
loss(x::Number, y::Number, lF::AbsoluteLoss) = abs(x - y)
loss(x::Number, y::Number, lF::MinkowskiLoss) = abs(x - y)^lF.p
loss(x::Number, y::Number, lF::SquaredLogLoss) = (log(x) - log(y))^2
loss(x::Number, y::Number, lF::AbsoluteLogLoss) = abs(log(x) - log(y))
loss(x::Number, y::Number, lF::SquaredPropLoss) = (x/y - 1)^2
loss(x::Number, y::Number, lF::AbsolutePropLoss) = abs(x/y - 1)
loss(x::Number, y::Number, lF::QLIKE) = x/y - log(x/y) - 1
loss(x::Number, y::Number, lF::QLIKEReverse) = y/x - log(y/x) - 1
function loss(x::Number, y::Number, lF::HuberLoss)
	e = x - y
	if abs(e) <= lF.p
		return(0.5 * e^2)
	else
		return(lF.p * (abs(e) - 0.5 * lF.p))
	end
end
#------ General methods that work for all loss functions
#Vector inputs (deliberately require same input type for x and y)
function loss{T<:Number}(x::Vector{T}, y::Vector{T}, lF::LossFunction)
	length(x) != length(y) && error("Input vectors must have matching length")
	return([ loss(x[n], y[n], lF) for n = 1:length(x) ])
end
#Matrix inputs (deliberately require same input type for x and y)
function loss{T<:Number}(x::Matrix{T}, y::Matrix{T}, lF::LossFunction)
	size(x, 1) != size(y, 1) && error("Input matrices must have matching number of rows")
	size(x, 2) != size(y, 2) && error("Input matrices must have matching number of columns")
	return([ loss(x[n, m], y[n, m], lF) for n = 1:size(x, 1), m = 1:size(x, 2) ])
end
#Vector and matrix input (deliberately require same input type for x and y)
function loss{T<:Number}(x::Vector{T}, y::Matrix{T}, lF::LossFunction)
	length(x) != size(y, 1) && error("Input vector must have same number of rows as input matrix")
	return([ loss(x[n], y[n, m], lF) for n = 1:size(y, 1), m = 1:size(y, 2) ])
end
#Reverse Vector and matrix input (note, separate function needed in case of asymmetric loss function input, e.g. L(x, y) != L(y, x))
function loss{T<:Number}(x::Matrix{T}, y::Vector{T}, lF::LossFunction)
	size(x, 1) != length(y) && error("Input vector must have same number of rows as input matrix")
	return([ loss(x[n, m], y[n], lF) for n = 1:size(x, 1), m = 1:size(x, 2) ])
end
lossdiff{T<:Number}(x1::T, x2::T, xBase::T, lF::LossFunction) = loss(x1, xBase, lF) - loss(x2, xBase, lF)
function lossdiff{T<:Number}(x1::Vector{T}, x2::Vector{T}, xBase::Vector{T}, lF::LossFunction)
	!(length(xBase) == length(x1) == length(x2)) && error("Input vectors must have matching length")
	return([ lossdiff(x1[n], x2[n], xBase[n], lF) for n = 1:length(xBase) ])
end
function lossdiff{T<:Number}(x1::Matrix{T}, x2::Vector{T}, xBase::Vector{T}, lF::LossFunction)
	!(length(xBase) == size(x1, 1) == length(x2)) && error("Input vectors must have matching length")
	size(x1, 2) < 1 && error("Input matrix is empty")
	ld = Array(T, size(x1, 1), size(x1, 2))
	for j = 1:size(x1, 2)
		for n = 1:size(x1, 1)
			ld[n, j] = lossdiff(x1[n, j], x2[n], xBase[n], lF)
		end
	end
	return(ld)
end
lossdiff{T<:Number}(x1::Vector{T}, x2::Matrix{T}, xBase::Vector{T}, lF::LossFunction) = -1 * lossdiff(x2, x1, xBase, lF) #asymmetry not a problem here




end # module
