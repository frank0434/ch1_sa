# try one factor at a time method 
library(ggplot2)
library(deSolve)
library(data.table)
source("ode_v2.R")

# values from papaer
# Y1(0) 0.9 Fixed Initial condition of Y1 
# Y2(0) 0.1 Free Initial condition of Y2 
# γ 1. Fixed Conversion factor - use y in R
# κ 1. Fixed Prey carrying capacity use k in R 
# h 0.9 Free Predator mortality rate 
# ζ 0.5 Free Allee threshold (prey) density use z in R
# H - - Prey harvesting rate (Eqn. 1.1)

# Create a data.table to store all the values for OFAT
# sample 20 points along 0 and 1 for each free parameters 
DT <- data.table(z = seq(0, 1, by = 0.05), k = 1, y = 1, 
                 h = seq(0, 1, by = 0.05),
                 Y2 = seq(0, 1, by = 0.05))
DT[, ':='(Y1 = 0.9,
          id = 1:.N)]
t <- seq(1, 500, by = 1)
l <- vector("list", length = nrow(DT))
# LOOP OVER EACH ROW 
for (i in 1:nrow(DT)) {
  print(i)
  allvals <- DT[id == i] |> unlist() 
  parameters <- allvals[c("z", "k","y","h")]
  states <- allvals[c("Y1","Y2")]
  sol <- ode(y = states, times = t, func = DetectPoints, parms = parameters)
  l[[i]] <- as.data.table(sol, keep.rownames = FALSE)
  names(l[i]) <- as.character(i)
  
}

ofat_dt <- data.table::rbindlist(l, use.names = TRUE, idcol = "id")
combined <- merge.data.table(ofat_dt[, .SD[.N], by = .(id)], DT, by = "id", suffixes = c(".post500",".pre"))
long <- combined[,.(Y2.post500, h, z, Y1.pre,id)] |> 
  melt(id.vars = c("id", "Y2.post500"))
long |> 
  ggplot(aes(value, Y2.post500))+
  geom_point() +
  facet_wrap(~ variable)
# the sampling method probably incorrect. 
# let's try a professional sampling tool ------------------
library(sensitivity)
paramlist <- parameterSets(list(z = c(0.1, 1), h = c(0.2, 0.95), Y2 = c(0.05, 0.2)),
              samples = 10, method = "grid") |> 
  as.data.table()
dim(paramlist)

paramlist[, ':='(id = 1:.N)]
t <- seq(1, 500, by = 1)
l <- vector("list", length = nrow(paramlist))
# LOOP OVER EACH ROW 
for (i in 1:nrow(paramlist)) {
  cat(i, " combination\n")
  allvals <- paramlist[id == i] |> unlist() 
  # constant 
  parameters <- c(allvals[c("z", "h")], k = 1, y = 1)
  states <- c(allvals[c("Y2")], 
              Y1 = 0.9)
  sol <- ode(y = states, times = t, func = DetectPoints, parms = parameters)
  l[[i]] <- as.data.table(sol, keep.rownames = FALSE)
  names(l[i]) <- as.character(i)
  
}
ofat_dt[id== 1, .SD[.N]]

ofat_dt <- data.table::rbindlist(l, use.names = TRUE, idcol = "id")
combined <- merge.data.table(ofat_dt[, .SD[.N], by = .(id)], paramlist, by = "id",
                             suffixes = c(".post500",".pre"))
long <- combined[,.(Y2.post500, h, z, Y2.pre,id)] |> 
  melt(id.vars = c("id", "Y2.post500"))
long |> 
  ggplot(aes(value, Y2.post500))+
  geom_point() +
  facet_wrap(~ variable)


# try just a morris? ------------------------------------------------------
library(ODEsensitivity)
set.seed(7292)
DetectPoints_v2 <- function(Time, State, Pars) {
  with(as.list(c(State, Pars)), {
    dY1 <- Y1 * (Y1 - z) * (1 - Y1) - Y1 * Y2
    dY2 <- 1 * (Y1 - h) * Y2
    return(list(c(dY1, dY2)))
  })
}
LVres_morris <- ODEmorris(mod = DetectPoints_v2,
                          pars = c("z", "h"),
                          state_init = c(Y1 = 0.9, Y2 = 0.1),
                          times =  seq(1, 500, by = 1),
                          binf = c(0.1, 0.5),
                          bsup = c(0.6, 0.9),
                          r = 500,
                          design = list(type = "oat",
                                        levels = 10, grid.jump = 1),
                          scale = TRUE,
                          ode_method = "lsoda",
                          parallel_eval = TRUE,
                          parallel_eval_ncores = parallel::detectCores())

str(LVres_morris)
plot(LVres_morris,state_plot = "Y2",pars_plot = "z")


sa_res <- LVres_morris$Y2 |> 
  t() |> 
  as.data.table()

sa_res |> 
  ggplot(aes(mu.star_z, y = mu.star_Y2 ))+
  geom_point()
