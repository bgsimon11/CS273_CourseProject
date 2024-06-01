library(reticulate)
use_python('/share/software/user/open/python/3.6.1/bin/python3.6', required=TRUE)
library(tensorflow)
keras <- import("tensorflow.keras")
library(keras)

library(abind)
library(Rtsne)
library(ggplot2)
library(magrittr)

physical_devices = tf$config$list_physical_devices('GPU')
print(physical_devices)

#print(tf$version$VERSION)

gen_path <- '/oak/stanford/groups/ccurtis2/users/bgsimon/CS273_CourseProject/CS273_CourseProject-main/Inference/generator'
disc_path <- '/oak/stanford/groups/ccurtis2/users/bgsimon/CS273_CourseProject/CS273_CourseProject-main/Inference/discriminator'

loaded_gen <- tf$keras$models$load_model(gen_path)
loaded_disc <- tf$keras$models$load_model(disc_path)

#print("GENERATOR")
#print(summary(loaded_gen))
#print("DISCRIMINATOR")
#print(summary(loaded_disc))

# data
library_size = 1e6
#validation_per = 0.15
n_genes = 19797L

# architecture
widths = c(256, 512, 1024)
latent_dim = 128L
embedding_dim = 64L
activation = "swish"
dropout_rate = 0.1

# optimization
epochs = 10
batch_size = 64L
learning_rate = 1e-4
discriminator_extra_steps = 5
gp_weight = 10.0
warmup_epochs = 10
ema = 0.999

#Data pipeline
#load("data.Rdata")
data_full <- t(read.csv('/oak/stanford/groups/ccurtis2/users/bgsimon/CS273_CourseProject/CS273_CourseProject-main/normalized_GSE152431_rna_raw_counts.csv', row.names = 1, check.names = FALSE))
print("Read in data successfully")

train_list_pre <- c("BM_24", "BM_43", "BM_47", "BM_51", "BM_54", "BM_55", "BM_64", "PB_64", "BM_65", "BM_72", "BM_80", "BM_82", "BM_88", "PB_88", "BM_90", "PB_91", "BM_97", "PB_97", "BM_106", "BM_122", "BM_133", "BM_141", "BM_143", "BM_150", "BM_156", "NNN_39", "NNN_40", "NNN_69", "NNN_68", "NNN_7")
train_list_post <- c("BM_25", "BM_45", "BM_48", "BM_53", "BM_58", "BM_57", "BM_66", "BM_67", "BM_76", "BM_83", "PB_83", "BM_84", "PB_84", "BM_89", "BM_92", "PB_92", "BM_95", "BM_98", "BM_107", "BM_126", "BM_135", "PB_135", "BM_142", "BM_144", "BM_152", "PB_152", "BM_157", "NNN_47", "NNN_42", "NNN_72", "NNN_71", "NNN_8")
val_list_pre <- c("BM_128", "PB_37", "BM_1", "PB_156", "PB_5", "BM_113", "PB_118", "BM_36", "PB_91")
val_list_post <- c("BM_130", "PB_38", "BM_3", "PB_157", "PB_6", "BM_114", "PB_120", "PB_38", "PB_93")
data <- data_full[rownames(data_full) %in% train_list_post,]
#print(nrow(data))

data <- data[,1:19797]

#print("done 1")

#n_samples <- dim(data)[1]
n_samples <- 32
n_validation <- 9
#sampled <- tf$random$shuffle(1:n_samples, seed = 1919810L)[1:n_validation] %>% as.integer()

x_data <- data
#max_tokens <- length(unique(labels$primary_site)) + length(unique(labels$sample_type))
#primary_site = "Bone Marrow", sample_type = "Post treatment Blood Cancer - Bone Marrow"
max_tokens <- 2
#y_data <- labels[,c("primary_site", "sample_type")]
#y_data <- rep(c("Bone Marrow", "Post treatment Blood Cancer - Bone Marrow"), n_samples)
y_data <- cbind(rep(c("Bone Marrow"), n_samples), rep(c("Post treatment Blood Cancer - Bone Marrow"), n_samples))
#rownames(y_data) <- labels$sample

colnames(y_data) <- c("primaty_site", "sample_type")

#print(y_data)

#print("done 2")

# training set
#x_train <- x_data[-sampled,]
x_train <- x_data
#x_train <- data.frame(lapply(x_train, as.numeric))
#y_train <- y_data[-sampled,]
y_train <- y_data


#print("x_train")
#print(x_train[1:5,1:5])
#print(head(rownames(x_train)))
#print(head(colnames(x_train)))
#print(tf$reduce_min(x_train, axis = 0L, keepdims = T))

# val set - Do we need?
x_val <- x_train
y_val <- y_train
#rm(x_data,y_data,data);gc()

#print("done 3")

#Define blocks
ResidualBlock_G <- function(x, width, dropout_rate){
  input_width <- x$shape[[2]]
  if(input_width == width){
    residual <- x
  } else {
    residual <- x %>% 
      layer_dense(units = width)
  }
  
  x <- x %>% 
    layer_batch_normalization(epsilon = 1e-5) %>% 
    layer_activation(activation = activation) %>% 
    layer_dense(units = width)
  
  x <- x %>% 
    layer_batch_normalization(epsilon = 1e-5) %>%
    layer_activation(activation = activation) %>% 
    layer_dropout(rate = dropout_rate) %>% 
    layer_dense(units = width)
  
  x <- layer_add(x, residual)
  return(x)
}

ResidualBlock_D <- function(x, width, dropout_rate){
  input_width <- x$shape[[2]]
  if(input_width == width){
    residual <- x
  } else {
    residual <- x %>% 
      layer_dense(units = width)
  }

  x <- x %>% 
    layer_activation(activation = activation) %>% 
    layer_dense(units = width)
    
  x <- x %>% 
    layer_activation(activation = activation) %>% 
    layer_dropout(rate = dropout_rate) %>% 
    layer_dense(units = width)
  
  x <- layer_add(x, residual)
  return(x)
}

#print("done 4")

#Build gen
build_generator <- function() {
  input <- layer_input(shape = c(latent_dim), name = "input")
  cls_input <- layer_input(shape = c(2), name = "cls_input", dtype = "int64")
  pre_treatment_rna_input <- layer_input(shape = c(n_genes), name = "pre_treatment_rna_input")
  
  cls_emb <- cls_input %>% 
    layer_embedding(input_dim = max_tokens + 2, output_dim = embedding_dim, input_length = 2, mask_zero = FALSE, trainable = TRUE) %>% 
    layer_flatten() %>% 
    layer_dense(units = embedding_dim * 2) %>% 
    layer_activation(activation = activation) %>% 
    layer_dense(units = latent_dim, name = "cls_emb")
  
  pre_treatment_rna_emb <- pre_treatment_rna_input %>% 
    layer_dense(units = latent_dim, activation = "relu", trainable = TRUE)
  
  backbone_in <- tf$math$multiply(input, cls_emb)
  backbone_in <- tf$math$multiply(backbone_in, pre_treatment_rna_emb)
  
  backbone_out <- backbone_in %>%
    layer_dense(units = widths[1]) %>% 
    ResidualBlock_G(width = widths[1], dropout_rate = dropout_rate) %>%
    ResidualBlock_G(width = widths[2], dropout_rate = dropout_rate) %>%
    ResidualBlock_G(width = widths[3], dropout_rate = dropout_rate)
  
  output <- backbone_out %>% 
    layer_batch_normalization(epsilon = 1e-5) %>% 
    layer_activation(activation = activation) %>% 
    layer_dense(units = n_genes, activation = "sigmoid")

  generator_model <- keras_model(list(input, cls_input, pre_treatment_rna_input), output, name = "generator")
  return(generator_model)
}


#print("done 5")

#Build disc
build_discriminator <- function() {
  input <- layer_input(shape = c(n_genes), name = "input")
  cls_input <- layer_input(shape = c(2), name = "cls_input", dtype = "int64")
  pre_treatment_rna_input <- layer_input(shape = c(n_genes), name = "pre_treatment_rna_input")
  
  cls_emb <- cls_input %>% 
    layer_embedding(input_dim = max_tokens + 2, output_dim = embedding_dim, input_length = 2, mask_zero = FALSE, trainable = TRUE) %>% 
    layer_flatten(name = "cls_emb")
  
  pre_treatment_rna_emb <- pre_treatment_rna_input %>% 
    layer_dense(units = latent_dim, activation = "relu", trainable = TRUE)
  
  backbone_in <- layer_concatenate(list(input, cls_emb, pre_treatment_rna_emb))
  
  backbone_out <- backbone_in %>%
    layer_dense(units = widths[3]) %>%
    ResidualBlock_D(width = widths[3], dropout_rate = dropout_rate) %>%
    ResidualBlock_D(width = widths[2], dropout_rate = dropout_rate) %>%
    ResidualBlock_D(width = widths[1], dropout_rate = dropout_rate)
    
  output <- backbone_out %>%
    layer_activation(activation = activation) %>%
    layer_dense(units = 1)
  
  discriminator_model <- keras_model(list(input, cls_input, pre_treatment_rna_input), output, name = "discriminator")
  return(discriminator_model)
}


#print("done 6")

#Callbacks
gan_monitor <- new_callback_class(  
  "gan_monitor",
  initialize = function(num_plot_samples) {
    self$num_plot_samples <- num_plot_samples
    if (!fs::dir_exists("saved_model(min-max)")) fs::dir_create("saved_model(min-max)")
    if (!fs::dir_exists("t-SNE plots(min-max)")) fs::dir_create("t-SNE plots(min-max)")
  },
  on_epoch_end = function(epoch, logs){
    if((epoch + 1) %% 100000 == 0){
      ############from train set############
      # random sampling from training set
      sampled <- sample(1:nrow(x_train), self$num_plot_samples)
    
      # real samples
      real_samples <- x_train[sampled,]
    
      # generated samples
      samples_labels <- y_train[sampled,]
      generated_samples <- model$generate(self$num_plot_samples, samples_labels) %>% as.matrix()
    
      # t-SNE
      labels <- c(rep("real",self$num_plot_samples),rep("gen",self$num_plot_samples))
      res.tsne <- rbind(real_samples, generated_samples) %>% Rtsne(check_duplicates = F)
      plot.tsne <- data.frame(res.tsne$Y,labels) 
      colnames(plot.tsne) <- c("X","Y","Label")
  
      ggplot(plot.tsne)+
        geom_point(aes(X,Y,color = Label,fill = Label))+
        scale_color_manual(values = c("red","grey"))+
        labs(x="",y="",title=paste0("t-SNE Plot(Epoch ",epoch + 1,")"))+
        theme_bw(base_size = 12)
      ggsave(paste0("t-SNE plots(min-max)/train t-SNE Plot(Epoch ",epoch + 1,").png"),width = 6,height = 4)

      ############from validation set############
      # real_samples
      real_samples <- x_val
      
      # generated samples
      samples_labels <- y_val
      generated_samples <- model$generate(nrow(real_samples), samples_labels) %>% as.matrix()
    
      # t-SNE
      labels <- c(rep("real",nrow(real_samples)),rep("gen",nrow(real_samples)))
      res.tsne <- rbind(real_samples, generated_samples) %>% Rtsne(check_duplicates = F)
      plot.tsne <- data.frame(res.tsne$Y,labels) 
      colnames(plot.tsne) <- c("X","Y","Label")
  
      ggplot(plot.tsne)+
        geom_point(aes(X,Y,color = Label,fill = Label))+
        scale_color_manual(values = c("red","grey"))+
        labs(x="",y="",title=paste0("t-SNE Plot(Epoch ",epoch + 1,")"))+
        theme_bw(base_size = 12)
      ggsave(paste0("t-SNE plots(min-max)/val t-SNE Plot(Epoch ",epoch + 1,").png"),width = 6,height = 4)
    }
    
    if((epoch + 1) %% 1 == 0){
      save_model_tf(model$discriminator,filepath = 
                    sprintf("./saved_model(min-max)/discriminator(Epoch %1i)",epoch + 1))
    }
    if((epoch + 1) %% 1 == 0){
      save_model_tf(model$ema_generator,filepath = 
                    sprintf("./saved_model(min-max)/ema_generator(Epoch %1i)",epoch + 1))
    }
  } 
)

#print("done 7")

warmup_decayed_lr <- new_learning_rate_schedule_class(
  "warmup_decayed_lr",
  initialize = function(warmup_steps,
                        decay_steps,
                        alpha = 0.0,
                        initial_learning_rate){
    super$initialize()
    self$warmup_steps <- warmup_steps
    self$decay_steps <- decay_steps
    self$alpha <- alpha
    self$initial_learning_rate <- initial_learning_rate
  },
  
  call = function(step){
    global_step <- tf$cast(step, tf$float32)
    warmup_lr <- (self$initial_learning_rate / self$warmup_steps) * global_step
    
    step <- global_step - self$warmup_steps
    step <- tf$math$minimum(step, self$decay_steps)
    cosine_decay <- 0.5 * (1.0 + tf$math$cos(pi * step / self$decay_steps))
    decayed <- (1.0 - self$alpha) * cosine_decay + self$alpha
    decay_lr<- self$initial_learning_rate * decayed
    lr <- tf$cond(global_step <= self$warmup_steps,
                  function() warmup_lr,
                  function() decay_lr)
    return(lr)
  }
)

#print("done 8")

#Build model
wgan_gp <- new_model_class(
  "wgan_gp",
  initialize = function(discriminator, 
                        generator,
                        discriminator_extra_steps = 5,
                        gp_weight = 10.0,
                        min, max) {
    super$initialize()
    self$discriminator <- discriminator
    self$generator <- generator
    self$ema_generator <- keras$models$clone_model(self$generator)
    self$ema_generator$set_weights(self$generator$get_weights())
    self$d_steps <- discriminator_extra_steps
    self$gp_weight <- gp_weight
    self$min <- min
    self$max <- max
    
    preprocessing <- import("tensorflow.keras.layers.experimental.preprocessing") #new
    
    #self$text_vectorization <- layer_text_vectorization(max_tokens = max_tokens + 2, # 2 contain "" and "[UNK]"
    #                                                    standardize = NULL,
    #                                                    split = function(x){tf$strings$split(x, sep = "/")},
    #                                                    output_mode = "int",
    #                                                    output_sequence_length = 2,
    #                                                    name = "text_vectorization")
    #self$text_vectorization$adapt(batch_size = batch_size, 
    #                              paste0(c(y_train, y_val)$primary_site, "/", c(y_train, y_val)$sample_type) %>% as_tensor())
    
    self$text_vectorization <- preprocessing$TextVectorization(
      max_tokens = as.integer(max_tokens + 2), # Ensure this is an integer
      standardize = NULL,
      split = function(x){tf$strings$split(x, sep = "/")},
      output_mode = "int",
      output_sequence_length = as.integer(2), # Explicitly set as integer
      name = "text_vectorization"
    )
    
    combined_data <- c(paste0(y_train[,1], "/", y_train[,2]), 
                       paste0(y_val[,1], "/", y_val[,2]))
    #print(combined_data)
    tensor_data <- tf$convert_to_tensor(combined_data, dtype = tf$string)
    #print("tensor_data")
    #print(tensor_data)
    self$text_vectorization$adapt(tensor_data)
    #print("hit1")
    # Adapt the layer
    #self$text_vectorization$adapt(
    #  #paste0(c(y_train, y_val)$primary_site, "/", c(y_train, y_val)$sample_type) %>% as_tensor()
    #  paste0("Bone Marrow", "/", "Post treatment Blood Cancer - Bone Marrow") %>% as_tensor()
    #)
 
  },
  
  compile = function(d_optimizer, g_optimizer, d_loss_fn, g_loss_fn) {
    super()$compile()
    self$d_optimizer <- d_optimizer
    self$g_optimizer <- g_optimizer
    self$d_loss_fn <- d_loss_fn
    self$g_loss_fn <- g_loss_fn
    self$d_loss_tracker <- tf$keras$metrics$Mean(name = "dis_loss")
    self$g_loss_tracker <- tf$keras$metrics$Mean(name = "gen_loss")
  }, 
  
  metrics = mark_active(function(){
    list(self$d_loss_tracker,
         self$g_loss_tracker)
  }),
  
  label2index = function(samples_labels){
    #samples_labels <- tf$strings$join(list(samples_labels$primary_site, samples_labels$sample_type), separator="/")
    #print("hit5")
    #samples_labels <- tf$strings$join(list("Bone Marrow", "Post treatment Blood Cancer - Bone Marrow"), separator="/")
    #samples_labels <- tf$convert_to_tensor(c("Bone Marrow/Post treatment Blood Cancer - Bone Marrow"))
    #samples_labels <- tf$convert_to_tensor(c("Bone Marrow/Post treatment Blood Cancer - Bone Marrow", 
    #                                     "Bone Marrow/Post treatment Blood Cancer - Bone Marrow"), dtype = tf$string)
    
    samples_labels <- tf$convert_to_tensor(rep(c("Bone Marrow/Post treatment Blood Cancer - Bone Marrow"),32), dtype = tf$string)
    
    #print(length(samples_labels))
    #print("hit6")
    #print(samples_labels)
    samples_labels <- self$text_vectorization(samples_labels)
    #print("hit2")
    #print(samples_labels)
    return(samples_labels)
  },
  
  denormalize = function(samples){
    samples <- samples * (self$max - self$min) + self$min
    sum <- tf$reduce_sum(samples, axis = -1L, keepdims = T)
    samples <- samples / sum * library_size
    return(samples)
  },
  
  normalizer = function(samples){
    samples <- tf$cast(samples, dtype = "float32")
    samples <- (samples - self$min) / (self$max - self$min)
    return(samples)
  },
  
  gradient_penalty = function(real_samples, fake_samples, samples_labels){
    batch_size <- tf$shape(real_samples)[1]
    alpha <- tf$random$normal(c(batch_size, 1L), 0.0, 1.0)
    diff <- fake_samples - real_samples
    interpolated <- real_samples + alpha * diff

    with(tf$GradientTape() %as% gp_tape, {
      gp_tape$watch(interpolated)
      # 1. Get the discriminator output for this interpolated samples.
      pred <- self$discriminator(list(interpolated, samples_labels), training=TRUE)
    }) 
    # 2. Calculate the gradients w.r.t to this interpolated samples
    grads <- gp_tape$gradient(pred, list(interpolated))[[1]]
    # 3. Calculate the norm of the gradients.
    norm <- tf$sqrt(tf$reduce_sum(tf$square(grads), axis= 1L))
    gp <- tf$reduce_mean((norm - 1.0) ^ 2)
    return(gp)
  },
  
  generate = function(num_samples, samples_labels){
    noise <- tf$random$normal(shape = c(num_samples, latent_dim))
    samples_labels <- self$label2index(samples_labels)
    generated_samples <- self$ema_generator(list(noise, samples_labels), training = FALSE)
    generated_samples <- self$denormalize(generated_samples)
    return(generated_samples)
  },
  
  train_step <- function(train_data) {
  c(real_samples, samples_labels, pre_treatment_rna_samples) %<-% train_data
  batch_size <- tf$shape(real_samples)[1]
  real_samples <- self$normalizer(real_samples)
  samples_labels <- self$label2index(samples_labels)
  pre_treatment_rna_samples <- self$normalizer(pre_treatment_rna_samples)
  
  # Train discriminator
  for(i in 1:self$d_steps) {
    noise <- tf$random$normal(shape = c(batch_size, latent_dim))
    with(tf$GradientTape() %as% tape, {
      # Generate fake samples from the latent vector
      fake_samples <- self$generator(list(noise, samples_labels, pre_treatment_rna_samples), training = TRUE) %>% 
        self$denormalize() %>% # to TPM
        self$normalizer() # to min-max
      # Get the logits for the fake samples
      fake_logits <- self$discriminator(list(fake_samples, samples_labels, pre_treatment_rna_samples), training = TRUE)
      # Get the logits for the real samples
      real_logits <- self$discriminator(list(real_samples, samples_labels, pre_treatment_rna_samples), training = TRUE)

      # Calculate the discriminator loss using the fake and real samples logits
      d_cost <- self$d_loss_fn(real_samp = real_logits, fake_samp = fake_logits)
      # Calculate the gradient penalty
      gp <- self$gradient_penalty(real_samples, fake_samples, samples_labels)
      # Add the gradient penalty to the original discriminator loss
      d_loss <- d_cost + gp * self$gp_weight
    })
    d_gradient <- tape$gradient(d_loss, self$discriminator$trainable_variables)
    # Update the weights of the discriminator using the discriminator optimizer
    self$d_optimizer$apply_gradients(
      zip_lists(d_gradient, self$discriminator$trainable_variables)
    )
  }
  
  # Train the generator
  noise <- tf$random$normal(shape = c(batch_size, latent_dim))
  with(tf$GradientTape() %as% tape, {
    # Generate fake samples using the generator
    generated_samples <- self$generator(list(noise, samples_labels, pre_treatment_rna_samples), training = TRUE) %>% 
      self$denormalize() %>% # to TPM
      self$normalizer() # to min-max
    # Get the discriminator logits for fake samples
    gen_samp_logits <- self$discriminator(list(generated_samples, samples_labels, pre_treatment_rna_samples), training = TRUE)
    # Calculate the generator loss
    g_loss <- self$g_loss_fn(gen_samp_logits) 
  })
  # Get the gradients w.r.t the generator  
  gen_gradient <- tape$gradient(g_loss, self$generator$trainable_variables)
  # Update the weights of the generator using the generator optimizer
  self$g_optimizer$apply_gradients(
    zip_lists(gen_gradient, self$generator$trainable_variables)
  )
  
  # EMA
  for(w in zip_lists(self$generator$weights, self$ema_generator$weights)) {
    w[[2]]$assign(self$ema * w[[2]] + (1 - self$ema) * w[[1]])
  }
  
  self$d_loss_tracker$update_state(d_cost)
  self$g_loss_tracker$update_state(g_loss)
  results <- list()
  for (m in self$metrics)
    results[[m$name]] <- m$result()
  results
  },
  
  test_step <- function(test_data) {
  c(real_samples, samples_labels, pre_treatment_rna_samples) %<-% test_data
  batch_size <- tf$shape(real_samples)[1]
  real_samples <- self$normalizer(real_samples)
  samples_labels <- self$label2index(samples_labels)
  pre_treatment_rna_samples <- self$normalizer(pre_treatment_rna_samples)
  
  noise <- tf$random$normal(shape = c(batch_size, latent_dim))
  # Generate samples and test
  fake_samples <- self$generator(list(noise, samples_labels, pre_treatment_rna_samples), training = FALSE) %>% 
    self$denormalize() %>% # to TPM
    self$normalizer() # to min-max
  fake_logits <- self$discriminator(list(fake_samples, samples_labels, pre_treatment_rna_samples), training = FALSE)
  real_logits <- self$discriminator(list(real_samples, samples_labels, pre_treatment_rna_samples), training = FALSE)
  d_loss <- self$d_loss_fn(real_samp = real_logits, fake_samp = fake_logits)
  g_loss <- self$g_loss_fn(fake_samp = fake_logits)
  
  self$d_loss_tracker$update_state(d_loss)
  self$g_loss_tracker$update_state(g_loss)
  results <- list()
  for (m in self$metrics)
    results[[m$name]] <- m$result()
  results
  }
)

#print("done 9")

#LR Schedule - might want to remove
D_lr_schedule <- warmup_decayed_lr(initial_learning_rate = learning_rate, 
                                   warmup_steps = ceiling(nrow(x_train)/batch_size) * warmup_epochs * discriminator_extra_steps,
                                   decay_steps = ceiling(nrow(x_train)/batch_size) * (epochs - warmup_epochs) * discriminator_extra_steps)
G_lr_schedule <- warmup_decayed_lr(initial_learning_rate = learning_rate, 
                                   warmup_steps = ceiling(nrow(x_train)/batch_size) * warmup_epochs,
                                   decay_steps = ceiling(nrow(x_train)/batch_size) * (epochs - warmup_epochs))

d_optimizer <- keras$optimizers$Adam(learning_rate = D_lr_schedule, beta_1 = 0.5, beta_2 = 0.9)
g_optimizer <- keras$optimizers$Adam(learning_rate = G_lr_schedule, beta_1 = 0.5, beta_2 = 0.9)

#print("done 10")

#Train
discriminator_loss <- function(real_samp, fake_samp){
  real_loss <- tf$reduce_mean(real_samp)
  fake_loss <- tf$reduce_mean(fake_samp)
  return(fake_loss - real_loss)
}

#print("done 11")
 
generator_loss <- function(fake_samp){
  return(-tf$reduce_mean(fake_samp))
}

#print("done 12")

model <- wgan_gp(
  discriminator = build_discriminator(),
  generator = build_generator(), 
  discriminator_extra_steps = discriminator_extra_steps,
  gp_weight = gp_weight,
  min = tf$reduce_min(x_train, axis = 0L, keepdims = T) %>% tf$cast(dtype = "float32"),
  max = tf$reduce_max(x_train, axis = 0L, keepdims = T) %>% tf$cast(dtype = "float32")
)

#print("done 13")

model %>% compile( 
  d_optimizer = d_optimizer,
  g_optimizer = g_optimizer,
  g_loss_fn = generator_loss,
  d_loss_fn = discriminator_loss
) 

#print("done 14")

#Fit
if (!fs::dir_exists("tf-logs(min-max)")) fs::dir_create("tf-logs(min-max)")
#tensorboard("tf-logs(min-max)")

#print("done 15")

if (!fs::dir_exists("checkpoints(min-max)")) fs::dir_create("checkpoints(min-max)")
checkpoint_filepath = "checkpoints(min-max)/model_weights"
model_checkpoint_callback <- callback_model_checkpoint(
  filepath = checkpoint_filepath,
  save_best_only = TRUE,
  save_weights_only = TRUE,
  monitor='dis_loss',
  mode='max',
  save_freq = 'epoch',
  period = NULL)
  
#print("done 16")

model %>% fit(x_train, y_train,
             epoch = epochs,
             batch_size = batch_size,
             validation_data = list(x_val, y_val),
             callbacks = list(gan_monitor(num_plot_samples = 1000L),
                              callback_tensorboard(log_dir = "tf-logs(min-max)",histogram_freq = 0),
                              model_checkpoint_callback)
             )
             
#print("done 17")
             
#Save
vocabulary <- model$text_vectorization$get_vocabulary()
min <- as.numeric(model$min)
max <- as.numeric(model$max)
Ensembl_ID <- colnames(x_train)
save(vocabulary, min, max, Ensembl_ID, max_tokens, n_genes, library_size, file = "other.Rdata")

print("done!")



