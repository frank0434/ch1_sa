library(data.table)
library(ggplot2)

data <- fread("data_raw/NL_Volkel_375.txt", skip = 53)
header <- fread("data_raw/NL_Volkel_375.txt", skip = 51, nrows = 1)
names(data) <- colnames(header)

data
# TN Minimum temperature 
# TX Maximum temperature 
# PG Daily mean sea level pressure  hPa
# FG Daily mean windspeed  m/s
# RH Daily precipitation amount mm
# Q Global radiation (in J/cm2)
# need IRRAD J/m2/day
# lat 51.6583231427806, lon 5.707457986917099
convertTom2 <- 10^4
hpatokpa <- 0.1

headernames <- "DAY	IRRAD	TMIN	TMAX	VAP	WIND	RAIN	SNOWDEPTH"
units <- "date	J/m2/day or hours	Celsius	Celsius	kPa	m/sec	mm	cm"
newdt <- data[,.(DAY = as.character(YYYYMMDD), IRRAD = Q, TMIN = TN, TMAX = TX, VAP = PG, 
        WIND = FG, RAIN = RH)]
newdt[, DAY := as.Date(DAY, format = "%Y%m%d")]
# Expt planting date
planting <- "2019-04-18"
harvesting <- "2019-09-25"
newdt <- newdt[(DAY >= planting) & (DAY <=harvesting)][, SNOWDEPTH := -999]
newdt[, ':='(IRRAD = IRRAD * convertTom2,
             VAP = VAP * hpatokpa,
             RAIN = RAIN/10)]
newdt
