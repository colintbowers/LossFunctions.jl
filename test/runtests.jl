

using LossFunctions


lossFuncVec = [SquaredError(), AbsoluteError(), MinkowskiError(), SquaredPropError(), AbsolutePropError(), SquaredLogError(), AbsoluteLogError(), QLIKE()]
posInputIdent = ["squaredLogError", "absoluteLogError", "QLIKE"]


#Deterministic tests (don't bother for expressions involving logs)
loss(2, 5, SquaredError()) != 9 && error("Invalid det 0")
loss(2, 5, AbsoluteError()) != 3 && error("Invalid det 0")
loss(2, 5, MinkowskiError()) != 9 && error("Invalid det 0")
loss(8, 2, SquaredPropError()) != 9 && error("Invalid det 0")
loss(8, 2, AbsolutePropError()) != 3 && error("Invalid det 0")
#Loop over loss functions and simulate input for each type
N = 10
M = 5
for k = 1:length(lossFuncVec)
	#k = 6
	#Get current loss function
	lF = lossFuncVec[k]
	#Simulate data
	x = randn()
	y = randn()
	xVec = randn(N)
	yVec = randn(N)
	xMat = randn(N, M)
	yMat = randn(N, M)
	if any(string(lF) .== posInputIdent)
		x = abs(x)
		y = abs(y)
		xVec = abs(xVec)
		yVec = abs(yVec)
		xMat = abs(xMat)
		yMat = abs(yMat)
	end
	#Test loss on simulated numbers (note, reverse order tested)
	xyOut = loss(x, y, lF)
	typeof(xyOut) != typeof(x) && error("Invalid sim 0")
	xyOut = loss(xVec, yVec, lF)
	length(xyOut) != N && error("Invalid sim 1")
	xyOut = loss(xMat, yMat, lF)
	size(xyOut, 1) != N && error("Invalid sim 2")
	size(xyOut, 2) != M && error("Invalid sim 3")
	xyOut = loss(xVec, yMat, lF)
	size(xyOut, 1) != N && error("Invalid sim 4")
	size(xyOut, 2) != M && error("Invalid sim 5")
	xyOut = loss(xMat, yVec, lF)
	size(xyOut, 1) != N && error("Invalid sim 6")
	size(xyOut, 2) != M && error("Invalid sim 7")
end




