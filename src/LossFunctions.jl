module LossFunctions
#-----------------------------------------------------------
#PURPOSE
#	Colin T. Bowers module for loss functions
#NOTES
#	To add a new loss function, provide a new type definition and any constructors under the TYPE DEFINTIIONS heading, then add an appropriate loss method for scalar inputs.
#	If you want to aggregate your loss series in some way, you should probably look at using the Distances package before thinking about this package
#LICENSE
#	MIT License (see github repository for more detail: https://github.com/colintbowers/DependentBootstrap.jl.git)
#-----------------------------------------------------------

#Load any entire modules that are needed (use import ModuleName1, ModuleName2, etc)

#Load any specific variables/functions that are needed (use import ModuleName1.FunctionName1, ModuleName2.FunctionName2, etc)
import 	Base.string,
		Base.show

#Specify the variables/functions to export (use export FunctionName1, FunctionName2, etc)
export 	LossFunction, #Abstract supertype
		SquaredError, #Loss function type
		AbsoluteError, #Loss function type
		MinkowskiError, #Loss function type
		SquaredLogError, #Loss function type
		AbsoluteLogError, #Loss function type
		SquaredPropError, #Loss function type
		AbsolutePropError, #Loss function type
		QLIKE, #Loss function type
		loss #Generic method for calculating loss between two things


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
type SquaredError <: LossFunction; end
type AbsoluteError <: LossFunction; end
type MinkowskiError <: LossFunction
	p::Real
	function MinkowskiError(p::Real)
		p <= 0 && error("Minkowski parameter must strictly positive")
		new(p)
	end
end
MinkowskiError() = MinkowskiError(2) #Default is SquaredError
type SquaredLogError <: LossFunction; end
type AbsoluteLogError <: LossFunction; end
type SquaredPropError <: LossFunction; end
type AbsolutePropError <: LossFunction; end
type QLIKE <: LossFunction; end
#---- string METHODS -------------
string(lF::SquaredError) = "squaredError"
string(lF::AbsoluteError) = "absoluteError"
string(lF::MinkowskiError) = "minkowskiError"
string(lF::SquaredLogError) = "squaredLogError"
string(lF::AbsoluteLogError) = "absoluteLogError"
string(lF::SquaredPropError) = "squaredProportionalError"
string(lF::AbsolutePropError) = "absoluteProportionalError"
string(lF::QLIKE) = "QLIKE"
#---- show METHODS -------------
show{T<:Union(SquaredError, AbsoluteError, SquaredLogError, AbsoluteLogError, SquaredPropError, AbsolutePropError, QLIKE)}(io::IO, lF::T) = println(io, "loss function = " * string(lF))
function show(io::IO, lf::MinkowskiError)
	println(io, "loss function = " * string(lF))
	println(io, "    p = " * string(lF.p))
end
show(lf::LossFunction) = show(STDOUT, lf)
#---- loss METHODS -------------
loss(x::Number, y::Number, lF::SquaredError) = (x - y)^2
loss(x::Number, y::Number, lF::AbsoluteError) = abs(x - y)
loss(x::Number, y::Number, lF::MinkowskiError) = abs(x - y)^lF.p
loss(x::Number, y::Number, lF::SquaredLogError) = (log(x) - log(y))^2
loss(x::Number, y::Number, lF::AbsoluteLogError) = abs(log(x) - log(y))
loss(x::Number, y::Number, lF::SquaredPropError) = (x/y - 1)^2
loss(x::Number, y::Number, lF::AbsolutePropError) = abs(x/y - 1)
loss(x::Number, y::Number, lF::QLIKE) = x/y - log(x/y) - 1
#---- METHODS FOR ARRAYS -------------
#Vector inputs (deliberately require same input type for x and y)
function loss{T<:Number}(x::Vector{T}, y::Vector{T}, lF::LossFunction)
	length(x) != length(y) && error("Input vectors must have matching length")
	xyOut = Array(T, length(x))
	for n = 1:length(x)
		xyOut[n] = loss(x[n], y[n], lF)
	end
	return(xyOut)
end
#Matrix inputs (deliberately require same input type for x and y)
function loss{T<:Number}(x::Matrix{T}, y::Matrix{T}, lF::LossFunction)
	size(x, 1) != size(y, 1) && error("Input matrices must have matching number of rows")
	size(x, 2) != size(y, 2) && error("Input matrices must have matching number of columns")
	xyOut = Array(T, size(x, 1), size(x, 2))
	for m = 1:size(x, 2)
		for n = 1:size(x, 1)
			xyOut[n, m] = loss(x[n, m], y[n, m], lF)
		end
	end
	return(xyOut)
end
#Vector and matrix input (deliberately require same input type for x and y)
function loss{T<:Number}(x::Vector{T}, y::Matrix{T}, lF::LossFunction)
	length(x) != size(y, 1) && error("Input vector must have same number of rows as input matrix")
	xyOut = Array(T, size(y, 1), size(y, 2))
	for m = 1:size(y, 2)
		for n = 1:size(y, 1)
			xyOut[n, m] = loss(x[n], y[n, m], lF)
		end
	end
	return(xyOut)
end
#Reverse Vector and matrix input (note, separate function needed in case of asymmetric loss function input, e.g. L(x, y) != L(y, x))
function loss{T<:Number}(x::Matrix{T}, y::Vector{T}, lF::LossFunction)
	size(x, 1) != length(y) && error("Input vector must have same number of rows as input matrix")
	xyOut = Array(T, size(x, 1), size(x, 2))
	for m = 1:size(x, 2)
		for n = 1:size(x, 1)
			xyOut[n, m] = loss(x[n, m], y[n], lF)
		end
	end
	return(xyOut)
end




end # module
