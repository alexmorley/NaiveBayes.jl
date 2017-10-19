export GaussianNB

# X: Feature vector as a p x n matrix
# c: Vector of classes
# k: Number of distinct classes
import Distributions
import ScikitLearnBase

type GaussianNB
    # The model hyperparameters (not learned from data)
    bias::Float64

    # The parameters learned from data
    mu::Matrix{Float64} # Matrix of cluster centers: p x k
    sigma::Matrix{Float64} # Diagonals of covariance matrices for each class
	p::Vector{Float64} # Vector of probabilities for each class
	counts::Matrix{Int}  
 
    # A constructor that accepts the hyperparameters as keyword arguments
    # with sensible defaults
    GaussianNB(; bias=0.0f0) = new(bias)
end

# This will define `clone`, `set_params!` and `get_params` for the model
ScikitLearnBase.@declare_hyperparameters(GaussianNB, [:bias])

# GaussianNB is a classifier
ScikitLearnBase.is_classifier(::GaussianNB) = true   # not required for transformers


function Distributions.rand(d::GaussianNB)
	p = size(d.mu, 1)
	c = Distributions.draw(d.drawtable)
	x = Array(Float64, p)
	for dim in 1:p
		x[dim] = rand(Normal(d.mu[dim, c], d.sigma[dim, c]))
	end
	x, c
end

function Distributions.rand(d::GaussianNB, n::Integer)
	p = size(d.mu, 1)
	X = Array(Float64, p, n)
	c = Array(Int, n)
	for obs in 1:n
		c[obs] = Distributions.draw(d.drawtable)
		for dim in 1:p
			X[dim, obs] = rand(Normal(d.mu[dim, c[obs]],
				                      d.sigma[dim, c[obs]]))
		end
	end
	X, c
end

function Distributions.mean(d::GaussianNB)
	p, c = size(d.mu)
	mx = zeros(Float64, p)
	for cl in 1:c
		mx += d.p[cl] * d.mu[:, c]
	end
	mx /= c
	mc = 0.0
	for cl in 1:c
		mc += d.p[cl] * cl
	end
	mx, mc
end


function ScikitLearnBase.fit!(model::GaussianNB, X::Matrix, c::Vector)
	p, n = size(X)
	nclasses = maximum(c)
	
	for i in 1:n
		model.mu[:, c[i]] += X[:, i]
		model.counts[c[i]] += 1
	end
	for cl in 1:nclasses
		model.mu[:, cl] /= model.counts[cl]
	end
	for i in 1:n
		model.sigma[:, c[i]] += (X[:, i] - model.mu[:, c[i]]).^2
	end
	for cl in 1:nclasses
		model.sigma[:, cl] = sqrt(model.sigma[:, cl] / (model.counts[cl] - 1)) + 1e-8
	end
	return model
end

function Distributions.logpdf(d::GaussianNB, x::Vector, c::Real)
	p = length(x)
	res = log(d.p[c])
	for dim in 1:p
		res += logpdf(Normal(d.mu[dim, c], d.sigma[dim, c]), x[dim])
	end
	return res
end

function Distributions.logpdf(d::GaussianNB, X::Matrix, c::Vector)
	p, n = size(X)
	res = zeros(Float64, n)
	for obs in 1:n
		res[obs] = log(d.p[c[obs]])
		for dim in 1:p
			res[obs] += logpdf(Normal(d.mu[dim, c[obs]],
				                      d.sigma[dim, c[obs]]),
			                   X[dim, obs])
		end
	end
	return res
end

function Distributions.loglikelihood(d::GaussianNB, X::Matrix, c::Vector)
	p, n = size(X)
	res = 0.0
	for obs in 1:n
		res += logpdf(d, X[:, obs], c[obs])
	end
	return res
end

function ScikitLearnBase.predict(d::GaussianNB, X::Matrix)
	nclasses = length(d.p)
	p, n = size(X)
	res = Array(Int, n)
	ll = Array(Float64, nclasses)
	for obs in 1:n
		for cl in 1:nclasses
			ll[cl] = logpdf(d, X[:, obs], cl)
		end
		res[obs] = indmax(ll)
	end
	return res
end
