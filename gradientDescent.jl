### A Pluto.jl notebook ###
# v0.14.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ f0906720-a4d5-426e-b5d9-190fc20209e9
begin
	using DelimitedFiles
	using Plots
end

# ╔═╡ d48bf640-a3e6-11eb-0d43-c1e0cfd320cb
md"
This notebook is an attempt to solve problems discussed in Machine Learning Course on Coursera
"

# ╔═╡ 93090f24-627b-4d3f-b603-abbd7425a095
md"
# **Programming Exercise 1 --**
"

# ╔═╡ e892c904-5eb6-45a0-8094-d947bbd012df
md"
## Linear Regression with one variable
"

# ╔═╡ a03aacf2-a6f6-4933-87db-b39ae18d75a5
md"
### Plotting the data
"

# ╔═╡ 07837f35-5662-4093-b2d5-616f8d4223a6
data = readdlm("ex1data1.txt", ',');

# ╔═╡ 33595998-b129-49f3-b1e0-82bd913103b4
X_old = data[:,1];

# ╔═╡ 4a48a847-3ac8-401e-8ef8-b9e1f6509a0d
y = data[:,2];

# ╔═╡ d5fd9c1e-4669-46ea-ab94-fbb8887e0dcf
scatter(X_old,y,xlabel = "Population of City in 10,000s",ylabel = "Profit in ₹10,000s",title = "Scatter Plot of Training Data")

# ╔═╡ 50279fde-a71c-46de-8b31-189262127385
md"
## Gradient Descent
"

# ╔═╡ c2e81215-0c9c-409f-b149-ab3a253caadb
md"
### Implementation
"

# ╔═╡ 6ae21439-2692-426b-b8e3-5e6207cca7d4
m = length(X_old); # number of training samples

# ╔═╡ bdd06b7b-12ef-44b2-bafd-e816bd40a9ee
X = hcat(ones(m,1), X_old); # adding column of ones

# ╔═╡ 650eee02-6307-42ce-9632-f7062075df70
theta = zeros(2,1);

# ╔═╡ d6f1a4cf-f24a-4dd3-b914-1644c8fc28d5
@bind alpha html"<input type = range max = 0.01 min = 0.0001 step = 0.001>"

# ╔═╡ 907cf6fb-0129-4392-9e3a-29917ae267e8
@bind iter html"<input type = range min = 200 max = 2000 step = 200>"

# ╔═╡ 27f9e814-c7b7-4a3a-93fb-9dd220a6861a
md"
The cost function for $iter itertation & learning rate of $alpha
"

# ╔═╡ e500aa32-291e-4c4e-92a8-cbc38acb537e
md"
### Prediction
"

# ╔═╡ a85eccd5-a887-4c0e-802c-cde76e7b2463
@bind population html"<input type = range min =  10000 max = 50000 step = 5000>"

# ╔═╡ 3059262e-3ee3-4363-bd8b-7ea0d658f065
predict = [1 (population/10000)];

# ╔═╡ 5b25240c-fad5-430e-b60e-4916fd779eee
md"
### Computing the cost J($\theta$)
"

# ╔═╡ 1a07332e-133a-4a1a-96c6-fe015173c1e2
function computeCost(X, y, theta)
	m = length(X)
	hypothesis = X * theta
	sqr_error = (hypothesis - y).^2
	cost = (0.5/m) * sum(sqr_error)
	return cost
end;

# ╔═╡ 77ef29ea-a707-405e-b8d3-f250de96c52b
md"
### Gradinet Descent Function
"

# ╔═╡ f521dde7-babc-4201-b16a-c15658474a10
function gradientDescent(X ,y, theta, alpha, iter)
	m = length(X)
	J_hist = zeros(iter,1)

	for i in 1:iter
		hypothesis = X * theta
		error = hypothesis - y
		change = X' * error
		theta = theta - ((alpha/m) * change)
		J_hist[i] = computeCost(X,y,theta)
	end

	return theta,J_hist
end;

# ╔═╡ 1ac794b2-e0e4-4a0f-a672-9a31e1e00e77
ans = gradientDescent(X ,y, theta, alpha, iter);

# ╔═╡ 3431bb97-35d1-4c6d-b5c5-b5ce7db6bb56
begin
	scatter(X_old,y,xlabel = "Population of City in 10,000s",ylabel = "Profit in    
		₹10,000s",title = "Plot of Training Data")
	plot!(X[:,2],X*ans[1])
end

# ╔═╡ 8d48b597-99a3-4dca-a513-339acf599f22
plot(1:iter,ans[2], xlabel = "No. of iters", ylabel  = "Cost of Function", title = "Cost vs Iters")

# ╔═╡ 765cdbe5-10a6-4ae2-9721-b4322944ae16
begin
	prediction = predict * ans[1] * 10000
	q = round(prediction[1,1])
end;

# ╔═╡ eb26c44c-60e9-4269-bb17-3650276eb1ef
md"
The cost of housing for $population population will be ₹ $q
"

# ╔═╡ Cell order:
# ╟─d48bf640-a3e6-11eb-0d43-c1e0cfd320cb
# ╟─93090f24-627b-4d3f-b603-abbd7425a095
# ╟─e892c904-5eb6-45a0-8094-d947bbd012df
# ╠═f0906720-a4d5-426e-b5d9-190fc20209e9
# ╟─a03aacf2-a6f6-4933-87db-b39ae18d75a5
# ╠═07837f35-5662-4093-b2d5-616f8d4223a6
# ╠═33595998-b129-49f3-b1e0-82bd913103b4
# ╠═4a48a847-3ac8-401e-8ef8-b9e1f6509a0d
# ╟─d5fd9c1e-4669-46ea-ab94-fbb8887e0dcf
# ╟─50279fde-a71c-46de-8b31-189262127385
# ╟─c2e81215-0c9c-409f-b149-ab3a253caadb
# ╠═6ae21439-2692-426b-b8e3-5e6207cca7d4
# ╠═bdd06b7b-12ef-44b2-bafd-e816bd40a9ee
# ╠═650eee02-6307-42ce-9632-f7062075df70
# ╟─d6f1a4cf-f24a-4dd3-b914-1644c8fc28d5
# ╟─907cf6fb-0129-4392-9e3a-29917ae267e8
# ╟─27f9e814-c7b7-4a3a-93fb-9dd220a6861a
# ╟─3431bb97-35d1-4c6d-b5c5-b5ce7db6bb56
# ╟─1ac794b2-e0e4-4a0f-a672-9a31e1e00e77
# ╟─8d48b597-99a3-4dca-a513-339acf599f22
# ╟─e500aa32-291e-4c4e-92a8-cbc38acb537e
# ╟─a85eccd5-a887-4c0e-802c-cde76e7b2463
# ╠═3059262e-3ee3-4363-bd8b-7ea0d658f065
# ╠═765cdbe5-10a6-4ae2-9721-b4322944ae16
# ╟─eb26c44c-60e9-4269-bb17-3650276eb1ef
# ╟─5b25240c-fad5-430e-b60e-4916fd779eee
# ╠═1a07332e-133a-4a1a-96c6-fe015173c1e2
# ╟─77ef29ea-a707-405e-b8d3-f250de96c52b
# ╠═f521dde7-babc-4201-b16a-c15658474a10
