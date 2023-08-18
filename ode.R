library(deSolve)
library(data.table)

# Define the system of ODEs
ode_func <- function(t, State, parameters) {
  with(as.list(c(State, parameters)), {
    
  # Y1 <- y[1]
  # Y2 <- y[2]
  # 
  # z <- parameters[1]
  # k <- parameters[2]
  # y <- parameters[3]
  # h <- parameters[4]
  # 
  dY1_dt <- Y1 * (Y1 - z) * (k - Y1) - Y1 * Y2
  dY2_dt <- y * (Y1 - h) * Y2
  
  return(list(c(dY1_dt, dY2_dt)))
  })
}

# Set initial conditions
initial_state <- c(Y1 = 1, Y2 = 0.1)

# Set parameter values
parameters1 <- c(z = 0.5, k = 1, y = 1, h = 0.9)


# Set the time points to solve the ODEs at
t <- seq(0, 150, by = 1)

# Solve the ODEs using ode()
sol <- ode(y = initial_state, times = t, func = ode_func, parms = parameters1)

oneset <- sol |> 
  as.data.table()
oneset <- oneset[, .(Y1,Y2, t(parameters1))
                 ][, t := t]
library(ggplot2)

oneset |> 
  ggplot(aes(x = t)) +
  geom_point(aes(y = Y1, color = "prey")) +
  geom_point(aes(y = Y2, color = "predator"))+
  ggtitle(parameters1)

parameters2 <- c(ζ = 0.1, κ = 1, γ = 0.8, h = 0.15)
parameters3 <- c(ζ = 0.01, κ = 1, γ = 0.8, h = 0.01)


library(pracma)

# chatgpt solutions - not working -----------------------------------------


# Define the function representing the system of derivatives
system_derivatives <- function(t, y, z, κ, γ, h) {
  with(as.list(c(y, z = z, κ = κ, γ = γ, h = h)), {
    dY1_dt <- Y1 * (Y1 - z) * (κ - Y1) - Y1 * Y2
    dY2_dt <- γ * (Y1 - h) * Y2
    
    return(c(dY1_dt, dY2_dt))
  })
}
parameters <- c(z = 0.1, k = 1, y = 0.8, h = 0.15)

# Set initial conditions
initial_state <- c(Y1 = 1, Y2 = 0.01)

# Set the time points to evaluate the derivatives
t <- seq(0, 100, by = 1)

# Calculate the partial derivatives
# partial_derivatives <- deriv(system_derivatives, "z")



# r pkg  ------------------------------------------------------------------
library(ODEsensitivity)
parameters
LVpars <- c("z", "h", "Y2", "k", "y")
LVbinf <- c(0,0, 0,1,1)
LVbsup <- c(1,1, 1,1,1)
# The initial values of the state variables:
LVinit <- c(Y1 = 1, Y2 = 0.1)
# The timepoints of interest:
LVtimes <- c(0.01, seq(1, 100, by = 1))
# Morris screening:
set.seed(7292)
# Warning: The following code might take very long!
LVres_morris <- ODEmorris(mod = ode_func,
                          pars = LVpars,
                          state_init = LVinit,
                          times = LVtimes,
                          binf = LVbinf,
                          bsup = LVbsup,
                          r = 500,
                          design = list(type = "oat",
                                        levels = 10, grid.jump = 1),
                          scale = TRUE,
                          ode_method = "lsoda",
                          parallel_eval = TRUE,
                          parallel_eval_ncores = 2)
plot(LVres_morris)
names(LVres_morris)
plot(LVres_morris, state_plot = "Y2")
sa_res <- LVres_morris$Y2 |> 
  t() |> 
  as.data.table()

sa_res |> 
  ggplot(aes(mu.star_z, y = mu.star_Y2 ))+
  geom_point()
