# Load necessary library
library(ggplot2)

# Create data for the Netherlands (NL), India (IND), and South Africa (SA)
nl_data <- data.frame(
  Month = 1:6,
  Temperature = c( 15, 18, 20, 22, 21, 18),
  Country = "Netherlands"
)

ind_data <- data.frame(
  Month = 1:3,
  Temperature = c(35, 20,  35),
  Country = "India"
)


create_plot <- function(data, color, month_label = c("Jan", "Jun"), title = "") {
  max_temp <- max(data$Temperature)
  min_temp <- min(data$Temperature)
  fs <- 24
  ggplot(data, aes(x = Month, y = Temperature)) +
    geom_smooth(method = "loess", se = FALSE, color = color, size = 3) +
    scale_x_continuous(breaks = range(data$Month), labels = month_label) +
    scale_y_continuous(breaks = max_temp, limits = c(min_temp, max_temp),
     labels = function(x) paste0(x, " °C")) +
    ggtitle(title) +
    theme_minimal() +
    theme(
      title = element_text(size = fs, face = "bold"),
      plot.title = element_text(hjust = 0.5),
      axis.title = element_blank(),
      axis.text.x = element_text(size = fs, face = "bold"),
      axis.text.y = element_text(size = fs, face = "bold"),
      panel.grid = element_blank(),
      axis.line = element_line(linewidth = 1.5),
      # axis.ticks = element_blank(),
      plot.margin = margin(5, 5, 5, 5, unit = "mm")
    )
}

# Create plots for each country
nl_plot <- create_plot(nl_data,  "red", month_label = c("May", "Sep"), title = "Temperate")
ind_plot <- create_plot(ind_data,  "red", month_label = c("Nov", "Feb"), title = "Subtropical")


# Print plots
print(nl_plot)
print(ind_plot)

ggsave("nl_temp_profile.png",plot = nl_plot, width = 4, height = 3, dpi = 300)
ggsave("ind_temp_profile.png",plot = ind_plot, width = 4, height = 3, dpi = 300)