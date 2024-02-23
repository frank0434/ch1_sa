# the ode for Bazykin-Berezovasakay model 
# Bazykin-berezovasakay model is built upon the Lotka-Volterra model by
# introducing predator hunting efficience and predator staturation
library(deSolve)

# Define the function that returns the derivatives
DetectPoints <- function(t, y, parms) {
  with(as.list(c(y, parms)), {
    dY1 <- Y1 * (Y1 - z) * (k - Y1) - Y1 * Y2
    dY2 <- y * (Y1 - h) * Y2
    return(list(c(dY1, dY2)))
  })
}

# # Define the parameters
# parms <- c(z = 0.5, k= 0.5, y = 0.5, h = 0.5)
# 
# # Define the initial values
# yini <- c(Y1 = 0.5, Y2 = 0.5)
# 
# # Define the time span
# times <- seq(0, 10, by = 0.01)
# 
# # Solve the ODEs
# out <- ode(y = yini, times = times, func = derivs, parms = parms)
# 
# # Plot the results
# plot(out)
