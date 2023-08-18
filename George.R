library(deSolve) ### https://cran.r-project.org/web/packages/deSolve/vignettes/deSolve.pdf
### Set vector with parameters and initial condition for simulation
parms1 <- c(r = 0.25, Mx = 10, M = 0.1, sr = 0, sMx = 0, sM0 = 1)
end.time <- 50
time.vec <- 0:(end.time)
### Define simulation model
Mod <- function(t, y, par) {
  with(as.list(c(par, y)), {
    dM <- r*M*(1 - M/Mx)
    dsr <- M*(1 - M/Mx) + r*(1 - (2*M)/Mx)*sr
    dsMx <- r*M^2/Mx^2 + r*(1 - (2*M)/Mx)*sMx
    dsM0 <- r*(1 - (2*M)/Mx)*sM0
    return(list(c(dM, dsr, dsMx, dsM0)))
  })
}
out.model <- ode(y = c(parms1["M"], parms1["sr"], parms1["sMx"], parms1["sM0"]),
                 times = time.vec, func = Mod, parms = c(parms1["r"], parms1["Mx"]) )
### Write data to file (for post-processing and pictures)
filename.output <- "out.dat"
write.table(out.model, filename.output)
### Plot
png(filename = "fig1AandB.png", width = 8, height = 4, units = 'in', res = 300)
par(mfrow=c(1,2))
plot(out.model[,1], out.model[,2], main = "Biomass logistic model", xlab = "time",
     ylab = "biomass", ylim = c(0,10), type = "l", lwd = 2, col = 3)
plot(out.model[,1], (out.model[,5]*0.04), main = "Local sensitivities", xlab = "time",
     ylab = "sensitivities", ylim = c(0,1.), type = "l", lwd = 2, col = 1)
lines(out.model[,1], (out.model[,3]*0.02), lwd = 2, col = 2)
lines(out.model[,1], (out.model[,4]*1), lwd = 2, col = 4)
dev.off()


# Temperature model --------------

T_L <- 18.85
T_H <- 29.85
T_AL <- 19726.85
T_AH <- 59726.85
x <- seq(2, 50, by = 1)
Tt <- (1 + exp(T_AL/x - T_AL/T_L) + exp(T_AH/T_H - T_AH/x)) ^ -1
plot(Tt)

Tmin <- 2
a <- log(2)/log((35 - Tmin)/(25-Tmin))
Teff <- (2 * (x-Tmin)^a * (25- Tmin)^a - (x - Tmin) ^ (2*a))/((25-Tmin)^ (2*a))
plot(Teff)


Mod <- function(t, x, y, par) {
  with(as.list(c(par, y)), {
    Teff <- (2 * (x-Tmin)^a * (25- Tmin)^a - (x - Tmin) ^ (2*a))/((25-Tmin)^ (2*a))
    
    dM <- r*M*(1 - M/Mx) * Teff
    
    dsr <- M*(1 - M/Mx) + r*(1 - (2*M)/Mx)*sr
    dsMx <- r*M^2/Mx^2 + r*(1 - (2*M)/Mx)*sMx
    dsM0 <- r*(1 - (2*M)/Mx)*sM0
    return(list(c(dM, dsr, dsMx, dsM0)))
  })
}
out.model <- ode(y = c(parms1["M"], parms1["sr"], parms1["sMx"], parms1["sM0"]),
                 times = time.vec, func = Mod, parms = c(parms1["r"], parms1["Mx"]) )
