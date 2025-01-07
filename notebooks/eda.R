library(tidyverse)
library(spatstat)

prefix <- "experiments/exp5/outputs/subject_1/sample_1/"
path <- "snapShots/cell_margin_0_242.csv"

df <- read_csv(paste0(prefix,path))

table(df$z)

df %>%
  mutate(Type=factor(Type)) %>%
  filter(z >=6 & z <= 6) %>%
  ggplot(aes(x,y,color=Type)) +
  geom_jitter(alpha=1) +
  # geom_point() +
  theme_classic()

time_points <- seq(1,1460,by=200)

df <- lapply(time_points,\(pt) {
  path <- paste0("snapShots/cell_core_0_",pt,".csv")
  df <- read_csv(paste0(prefix,path)) %>%
    mutate(Type=factor(Type)) %>%
    filter(z >=6 & z <= 6) %>%
    mutate(time = pt) %>%
    mutate(extra = as.character(extra))
  if(nrow(df) == 0) {
    return(NULL)
  }
  
  df
  
}) %>%
  purrr::compact() %>%
  bind_rows()

df %>%
  ggplot(aes(x,y,color=Type)) +
  geom_jitter(alpha=1) +
  # geom_point() +
  theme_classic() +
  facet_wrap(~time)

path <- "QSP_0.csv"
df <- read_csv(paste0(prefix,path))

cpts <- c("Tum","LN")

d2 <- df %>%
  mutate(across(-time,~ .x / (max(.x,na.rm = T) + .Machine$double.eps))) %>%
  pivot_longer(-time) %>%
  separate(name,c("compartment","type"),sep = "\\.")

types_exclude <- c("PD1","PDL1")

ps <- lapply(cpts,\(cpt) {
  p <- d2 %>%
    filter(compartment == cpt) %>%
    # filter(!str_detect(type, str_c(types_exclude, collapse = "|"))) %>%
    ggplot(aes(time,value,color=type)) +
    geom_line() +
    # facet_wrap(~compartment,nrow=3) +
    theme_classic() +
    scale_color_manual(values=as.vector(pals::glasbey())) +
    ggtitle(cpt) +
    guides(color = guide_legend(ncol=3)) +
    xlim(c(0,500))
})

patchwork::wrap_plots(ps,nrow = 4,ncol = 1)


# Multiple simulations
types_exclude <- c("PD1","PDL1")
lapply(1:5,\(i) {
  path <- paste0("exp1/vp/subject_1/sample_",i,"/Outputs/QSP_0.csv")
  df <- read_csv(path)
  
  cpts <- c("Tum")
  
  d2 <- df %>%
    mutate(across(-time,~ .x / (max(.x,na.rm = T) + .Machine$double.eps))) %>%
    pivot_longer(-time) %>%
    separate(name,c("compartment","type"),sep = "\\.") %>%
    filter(compartment %in% cpts) %>%
    # filter(!str_detect(type, str_c(types_exclude, collapse = "|"))) %>%
    mutate(sample = i)
  
}) %>%
  bind_rows() -> d2

d2 %>%
  mutate(time = time*0.25) %>%
  ggplot(aes(time,value,color=factor(sample))) +
  geom_line() +
  facet_wrap(~type) +
  theme_classic() +
  scale_color_manual(values=as.vector(pals::glasbey())) +
  # ggtitle(cpt) +
  # guides(color = guide_legend(ncol=3)) +
  xlim(c(0,500*0.25))

# smallest resolution for temproal sampling is every 3 days, only from lymph nodes and blood

# from tumor site itself (spatial info): only 

# D1_0: Total concentration of antigenic proteins from cancer cells
# C1: cancer cells
# Cp: competing proteins
# Teff_1_0: T cells reactive to cancer cells
# Ckine_Mat: Concentration of maturation cytokines in tumor extracelullar space
# C1_PDL1: Average number of free PDL1 molecules on Cancer cells in the tumor
# C1_PDL1_Teff_PD1: Number of PDL1-PD1 complexes between T cells and cancer cells
# n_clone_p1_0: number of clones of the generic antigen. It is highely cacner dependent and patient specific
# Kd_p1_0_M1: Binding affinity of peptide/MHC binding. It is antigen specific

# now let's look at spatial at different timepoints across samples
time_points <- seq(1,1460,by=100)

lapply(1:5,\(i) {
  lapply(time_points,\(pt) {
    path <- paste0("exp1/vp/subject_1/sample_",i,"/Outputs/snapShots/cell_margin_0_",pt,".csv")
    df <- read_csv(path) %>%
      mutate(sample = i) %>%
      mutate(time = pt)
  }) %>%
    bind_rows()
}) %>%
  bind_rows() %>%
  mutate(Type=factor(Type)) %>%
  filter(z >=6 & z <= 6) -> spdf

spdf %>%
  ggplot(aes(x,y,color=Type)) +
  geom_jitter(alpha=1) +
  # geom_point() +
  theme_classic() +
  facet_grid(sample~time)

rad <- seq(0,25,length.out=100)

gx_df <- spdf %>%
  filter(time == 101) %>%
  group_by(sample,time) %>%
  group_map(~{
    .x$Type <- droplevels(.x$Type)
    pat <- ppp(.x$x,.x$y,marks = .x$Type,window = square(c(0,25)))
    pat <- rjitter(pat,radius = 0.5)
    # plot(pat)
    gx <- alltypes(pat,Gcross)
      # as.data.frame() %>%
      # mutate(sample=.y$sample,
      #        time=.y$time)
    plot(gx,title=paste0("gx for sample ", .y$sample, " at time ", .y$time))
  })

