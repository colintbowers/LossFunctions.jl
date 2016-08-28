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
		SymmetricLossFunction, #Abstract subtypes
		AsymmetricLossFunction, #Abstract subtypes
		SquaredLoss,
		AbsoluteLoss,
		MinkowskiLoss,
		HuberLoss,
		SquaredLogLoss,
		AbsoluteLogLoss,
		SquaredPropLoss,
		AbsolutePropLoss,
		QLIKE,
		QLIKEReverse,
		loss, #Generic method for calculating loss, ie L(x, y), or L(e) for the case e = x - y
		lossdiff #Generic method for calculating a loss differential, ie L(x-BaseCase) - L(y-BaseCase)


#******************************************************************************

#----------------------------------------------------------
#SET CONSTANTS FOR MODULE
#----------------------------------------------------------
const singleInputLossMethods = Union{SquaredLoss, AbsoluteLoss, MinkowskiLoss, HuberLoss}::Union



#----------------------------------------------------------
#LOSS FUNCTION TYPES
#----------------------------------------------------------
#Abstract types
abstract LossFunction
abstract SymmetricLossFunction <: LossFunction
abstract AsymmetricLossFunction <: LossFunction
#Loss function types
type DummyLoss <: LossFunction ; end #dummy type that does nothing
type SquaredLoss <: SymmetricLossFunction ; end
type AbsoluteLoss <: SymmetricLossFunction ; end
type MinkowskiLoss <: SymmetricLossFunction
	p::Float64
	MinkowskiLoss(p::Float64) = (p <= 0) ? error("Minkowski parameter must be strictly positive") : new(p)
end
MinkowskiLoss() = MinkowskiLoss(2.0) #Default is SquaredLoss
MinkowskiLoss(p::Number) = MinkowskiLoss(convert(Float64, p))
type SquaredLogLoss <: SymmetricLossFunction ; end
type AbsoluteLogLoss <: SymmetricLossFunction ; end
type SquaredPropLoss <: AsymmetricLossFunction ; end
type AbsolutePropLoss <: AsymmetricLossFunction ; end
type QLIKE <: AsymmetricLossFunction ; end
type QLIKEReverse <: AsymmetricLossFunction ; end
type HuberLoss <: SymmetricLossFunction
	p::Float64
	HuberLoss(p::Float64) = (p <= 0) ? error("Huber loss parameter must be strictly positive") : new(p)
end
HuberLoss() = HuberLoss(1.0)
HuberLoss(p::Number) = HuberLoss(convert(Float64, p))

#----------------------------------------------------------
#LOSS FUNCTION ADMIN METHODS
#----------------------------------------------------------
#string
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
#show
function show{T<:Union{MinkowskiLoss, HuberLoss}}(io::IO, lf::T)
	println(io, "Loss function = " * string(lF))
	println(io, "    p = " * string(lF.p))
end
show(io::IO, lf::LossFunction) = println(io, "Loss function = " * string(lF))
show(lf::LossFunction) = show(STDOUT, lf)
#tolossfunction    WARNING: NOT TYPE STABLE
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
tolossfunction(x::String) = tolossfunction(symbol(x))


#----------------------------------------------------------
# CORE METHODS FOR COMPUTING LOSS
#----------------------------------------------------------
#Methods that can be defined in terms of e = x - y, i.e. single input loss methods
loss(e::Number, lF::SquaredLoss) = e^2
loss(e::Number, lF::AbsoluteLoss) = abs(e)
loss(e::Number, lF::MinkowskiLoss) = abs(e)^lF.p
loss(e::Number, lF::HuberLoss) = abs(e) <= lF.p ? 0.5*e^2 : lF.p*(abs(e)-0.5*lF.p)
#Methods that cannot be defined in terms of e = x - y
loss{T<:singleInputLossMethods}(x::Number, y::Number, lF::T) = loss(x - y)
loss(e::Number, lF::SquaredLogLoss) = (log(x) - log(y))^2
loss(e::Number, lF::AbsoluteLogLoss) = abs(log(x) - log(y))
loss(e::Number, lF::SquaredPropLoss) = (x/y - 1)^2
loss(e::Number, lF::AbsolutePropLoss) = abs(x/y - 1)
loss(e::Number, lF::QLIKE) = x/y - log(x/y) - 1
loss(e::Number, lF::QLIKEReverse) = y/x - log(y/x) - 1
#Method for a loss differential (used a lot in forecast evaluation), i.e. L(x, basecase) - L(y, basecase)
lossdiff{T<:Number}(x1::T, x2::T, xBase::T, lF::LossFunction) = loss(x1, xBase, lF) - loss(x2, xBase, lF)


#----------------------------------------------------------
# ARRAY BASED EXTENSIONS TO CORE METHODS
#----------------------------------------------------------
loss{T1<:Number, T2<:singleInputLossMethods}(e::Vector{T}, lF::LossFunction) = T[ loss(e[n], lF) for n = 1:length(e) ]
loss{T<:Number}(x::Vector{T}, y::Vector{T}, lF::LossFunction) = length(x) == length(y) ? T[ loss(x[n], y[n], lF) for n = 1:length(x) ] : error("Length mismatch")
loss{T<:Number}(x::Matrix{T}, y::Matrix{T}, lF::LossFunction) = size(x) == size(y) ? T[ loss(x[n, m], y[n, m], lF) for n = 1:size(x, 1), m = 1:size(x, 2) ] : error("size mismatch")
loss{T<:Number}(x::Vector{T}, y::Matrix{T}, lF::LossFunction) = length(x) == size(y, 1) ? T[ loss(x[n], y[n, m], lF) for n = 1:size(y, 1), m = 1:size(y, 2) ] : error("size mismatch")
loss{T<:Number}(x::Matrix{T}, y::Vector{T}, lF::LossFunction) = size(x, 1) == length(y) ? T[ loss(x[n, m], y[n], lF) for n = 1:size(x, 1), m = 1:size(x, 2) ] : error("Size mismatch")
lossdiff{T<:Number}(x1::Vector{T}, x2::Vector{T}, xBase::Vector{T}, lF::LossFunction) = (length(xBase) == length(x1) == length(x2) ? T[ lossdiff(x1[n], x2[n], xBase[n], lF) for n = 1:length(xBase) ] : error("Length mismatch")
lossdiff{T<:Number}(x1::Matrix{T}, x2::Vector{T}, xBase::Vector{T}, lF::LossFunction) = (length(xBase) == size(x1, 1) == length(x2)) ? T[ lossdiff(x1[n, m], x2[n], xBase[n], lF) for n = 1:size(x1, 1), m = 1:size(x1, 2) ]
lossdiff{T<:Number}(x1::Vector{T}, x2::Matrix{T}, xBase::Vector{T}, lF::LossFunction) = -1 * lossdiff(x2, x1, xBase, lF)




end # module
