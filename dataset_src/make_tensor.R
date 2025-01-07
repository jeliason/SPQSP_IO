library(tidyverse)
library(fastDummies)
library(reticulate)

min_dim <- 0
max_dim <- 24
nx <- ny <- 25

SYSTEM_ENV <- Sys.getenv("SYSTEM_ENV")

if(SYSTEM_ENV == "laptop") {
  use_condaenv("spqsp")
} else {
  use_virtualenv("~/virtual_envs/physicell")
}

# utils <- reticulate::import_from_path("utils","sims/src/")
np <- reticulate::import("numpy")

args <- commandArgs(trailingOnly = TRUE)
OUTPUT_PATH <- args[1]
sim_id <- args[2]

# OUTPUT_PATH <- "exp1/vp/subject_1/"

unique_types <- c(1,2,3)
unique_states <- c(0,3,4,5,6,7,8)
z_slices <- c(12)

make_array <- \(sim_id) {
  print(sim_id)
  cell_df_file_path <- paste0(OUTPUT_PATH,"/sample_",sim_id,"/snapShots")
  # get all files in path that start with "cells_core"
  cell_df_files <- list.files(cell_df_file_path, pattern="cell", full.names = FALSE)
  cells_df_list <- lapply(cell_df_files, \(file) {
    nm <- gsub(".csv","",file)
    spl <- str_split(nm,"_")[[1]]
    d <- read_csv(paste0(OUTPUT_PATH,"/sample_",sim_id,"/snapShots/",file)) %>%
      filter(z %in% z_slices) %>%
      select(-z) %>%
      # mutate(sim_id = sim_id) %>%
      select(-extra) %>%
      mutate(roi = spl[2],
             roi_id = spl[3],
             time = spl[4]) %>%
      mutate(roi = paste0(roi,"_",roi_id)) %>%
      select(-roi_id)
    if(nrow(d) == 0) return(NULL)
    
    d
  }) %>%
    purrr::compact()
  
  
  cells_df <- bind_rows(cells_df_list)

  
  cells_df <- cells_df %>%
    mutate(Type = factor(Type,levels=unique_types),
           State = factor(State,levels=unique_states))
  
  cells_df <- dummy_cols(cells_df, select_columns = c("Type","State"),remove_selected_columns = TRUE)
  
  cells_df
  
  timestep <- cells_df$time
  roi <- cells_df$roi
  # 
  # # Creating the pixel boundaries
  # x_bins <- seq(min_dim, max_dim, length.out = nx + 1)
  # y_bins <- seq(min_dim, max_dim, length.out = ny + 1)
  # 
  # # Assigning each point to a pixel
  # x_pixel <- cut(x, breaks = x_bins, labels = FALSE, include.lowest = TRUE)
  # y_pixel <- cut(y, breaks = y_bins, labels = FALSE, include.lowest = TRUE)
  
  # # Create a dataframe with the pixel and image information
  # data <- cells_df %>%
  #   mutate(x_pixel=x_pixel,
  #          y_pixel=y_pixel) %>%
  #   select(-c(position_x,position_y))
  
  # Counting the number of points of each type in each pixel for each image
  cells_df %>%
    pivot_longer(-c(x,y,time,roi)) %>%
    dplyr::group_by(roi,time, x, y, name) %>%
    dplyr::summarise(count = sum(value), .groups = 'drop') -> pixel_counts
  
  # Filling missing combinations with NA
  full_grid <- expand_grid(
    roi = unique(roi),
    time = unique(timestep),
    name = unique(pixel_counts$name),
    x = min_dim:max_dim,
    y = min_dim:max_dim
  )
  
  # Merge with the counts, ensuring missing combinations are NA
  final_df <- full_grid %>%
    dplyr::left_join(pixel_counts, by = c("roi","time", "x", "y","name")) %>%
    dplyr::arrange(roi,time, x, y,name) %>%
    mutate(count=ifelse(is.na(count),0,count))
  
  data_layers <- unique(final_df$name)
  timesteps <- unique(final_df$time)
  unique_roi <- unique(final_df$roi)
  
 arrs <-  lapply(unique_roi,\(ROI) {
    arrs <- lapply(timesteps,\(step) {
      mats <- lapply(data_layers,\(layer) {
        mat <- final_df %>%
          filter(time == step) %>%
          select(-time) %>%
          filter(name == layer) %>% 
          select(-name) %>%
          filter(roi == ROI) %>%
          select(-roi) %>%
          pivot_wider(names_from = x,values_from = count) %>%
          replace(is.na(.),0) %>%
          select(-y) %>%
          as.matrix()
      })
      
      arr <- simplify2array(mats)
    })
    arr <- simplify2array(arrs)
  })
  
  final_arr <- simplify2array(arrs)
  # dim(arr)
  # 
  # data_layers
  # image(arr[,,1,50])
  
  
  # conc_df <- read_parquet(conc_df_file)
  # conc_df <-  conc_df %>%
  #   pivot_longer(-c(mesh_center_m,mesh_center_n,timestep))
  # 
  # conc_layers <- unique(conc_df$name)
  # 
  # conc_arrs <- lapply(timesteps,\(step) {
  #   mats <- lapply(conc_layers,\(layer) {
  #     mat <- conc_df %>%
  #       filter(timestep == step) %>%
  #       select(-timestep) %>%
  #       filter(name == layer) %>% 
  #       select(-name) %>%
  #       pivot_wider(names_from = mesh_center_m,values_from = value) %>%
  #       replace(is.na(.),0) %>%
  #       select(-mesh_center_n) %>%
  #       as.matrix()
  #   })
  #   
  #   arr <- simplify2array(mats)
  # })
  # 
  # conc_arr <- simplify2array(conc_arrs)
  # 
  # final_arr <- abind::abind(arr,conc_arr,along=3)
}


# gzip <- reticulate::import("gzip")
# tensor_path <- paste0(OUTPUT_PATH,"data.npy.gz")
# np$save(tensor_path,final_arr)
# 
# f <- gzip$GzipFile(tensor_path, "w")
# np$save(file=f, arr=final_arr)
# f$close()
# 
# 
# f <- gzip$GzipFile(tensor_path, "r")
# final_arr2 <- np$load(f)
# f$close()
# 
# dim(final_arr2)
# 

final_arr <- make_array(sim_id)
final_arr_py <- r_to_py(final_arr)
np$savez_compressed(paste0(OUTPUT_PATH,"/sample_",sim_id,"/spatial.npz"),final_arr_py)


unlink(paste0(OUTPUT_PATH,"/sample_",sim_id,"/snapShots"), recursive = TRUE)
# Create a temporary directory for storing files
# dir.create("temp_simulation_data")
# 
# n_frames <- dim(final_arr)[4]
# 
# grid <- read_csv(paste0(OUTPUT_PATH,"grid.csv"))
# theta_id <- grid$theta_id[sim_id + 1]
# theta <- np$load(paste0(OUTPUT_PATH,"theta.npy"))
# nms <- names(utils$get_param_dict())
# 
# generative_params <- as.list(theta[theta_id,]) # DOUBLE CHECK ON THESE IDS!!!!!!! zero- vs one-based indexing
# names(generative_params) <- nms
# 
# # Save frames as .npy
# for (i in 1:n_frames) {
#   frame_filename <- sprintf("temp_simulation_data/frame_%04d.npy", i - 1)
#   arr <- np$array(r_to_py(final_arr[,,,i]), dtype = "float32")
#   
#   np$save(frame_filename, arr)
#   
#   param_filename <- sprintf("temp_simulation_data/frame_%04d.json", i - 1)
#   jsonlite::write_json(generative_params, param_filename)
# }
# 
# # Create a .tar archive
# output_tar <- sprintf("simulation_%06d.tar", sim_id)
# tar(output_tar, files = list.files("temp_simulation_data", full.names = TRUE), compression = "gzip")
# 
# # Clean up temporary files
# unlink("temp_simulation_data", recursive = TRUE)
# 
# cat("Saved simulation data to", output_tar, "\n")
