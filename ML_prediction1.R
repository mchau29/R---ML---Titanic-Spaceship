#-----------------------LIBRARY & READ-IN FILES-------------------------
library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(stringr)
library(RColorBrewer)

options(digits = 2)
setwd("C:/Users/minhk/OneDrive/Data Science stuffs/PORTFOLIO/ML - Titanic Spaceship")
set.seed(20)

train.full <- read.csv("train.csv")
unseen <- read.csv("test.csv")

#-------------------------------Train.clean-------------------------------------

#Train.clean = Split Passenger ID & Cabin into separate columns
train.clean <- train.full %>%
  separate(PassengerId, c("party.identifiers","ps"),"_") %>%
  mutate(party.size = as.numeric(ps)) %>%
  select(!ps)

train.clean <- train.clean %>%
  separate(Cabin,c("deck","NUMBER","SIDE"),"/") %>%
  mutate(side = ifelse(SIDE == "P","Port","Starboard")) %>%
  select(!c(NUMBER,SIDE))

# Indicating if using VR deck
train.clean <- train.clean %>%
  mutate(VR.use = ifelse(VRDeck >0,"yes","no"))
                         
which(colSums(is.na(train.clean))>0) # moneys, side, age, VR.use 

# Money related columns: assuming NA = 0 & create luxury col instead
RoomService = train.clean["RoomService"][is.na(train.clean["RoomService"])] <- 0
FoodCourt = train.clean["FoodCourt"][is.na(train.clean["FoodCourt"])] <- 0
ShoppingMall = train.clean["ShoppingMall"][is.na(train.clean["ShoppingMall"])] <- 0
Spa = train.clean["Spa"][is.na(train.clean["Spa"])] <- 0
VRDeck = train.clean["VRDeck"][is.na(train.clean["VRDeck"])] <- 0

train.clean <- train.clean %>%
  mutate(luxury = RoomService +FoodCourt +ShoppingMall +Spa +VRDeck)

# cryosleep 
train.clean %>%
  select(CryoSleep,
         luxury) %>%
  group_by(CryoSleep) %>%
  summarise(n = n(),
            avg = mean(luxury))

# people in cryosleep not spending money => luxury = 0 = true, luxury > 0 = false
train.clean <- train.clean %>%
  mutate(CryoSleep = ifelse(CryoSleep == "" & luxury ==0,"True",CryoSleep))

train.clean <- train.clean %>%
  mutate(CryoSleep = ifelse(CryoSleep == "" & luxury >0,"False",CryoSleep))

train.clean <- train.clean %>%
  mutate(VR.use = ifelse(CryoSleep == "True" &
                           VR.use == "unknown","no",VR.use))

# Putting "unknown" for categorical columns with " ":
train.clean <- train.clean %>%
  mutate(side = ifelse(is.na(side) ==TRUE,"unknown",side),
         VR.use = ifelse(is.na(VR.use) ==TRUE,"unknown", VR.use),
         HomePlanet = ifelse(HomePlanet == "","unknown",HomePlanet),
         Destination = ifelse(Destination == "","unknown",Destination),
         VIP = ifelse(VIP == "","unknown",VIP),
         deck = ifelse(deck == "","unknown",deck))

# Age: to make age easy to analyze, make age group instead 

train.clean <- train.clean %>%
  mutate(age.group = ifelse(Age >= 61, "elder",
                     ifelse(Age <=60 & Age >= 51, "adult.51-60",
                     ifelse(Age <=50 & Age >= 41,"adult.41-50",
                     ifelse(Age <=40 & Age >= 31,"adult.31-40",
                     ifelse(Age <=30 & Age >= 21,"adult.21-30",
                     ifelse(Age <=20 & Age >= 10,"teen","child")))))))

age.group1 <- train.clean %>%
  filter(CryoSleep == "False") %>%
  select(age.group,
         luxury) %>%
  group_by(age.group) %>%
  drop_na() %>%
  summarise(n = n(),
            avg = mean(luxury)) %>%
  arrange(desc(avg))
age.group1

train.clean <- train.clean %>%
  mutate(age.group = ifelse(is.na(age.group) == TRUE &
                            CryoSleep == "False" &
                            luxury >= age.group1$avg[
                              age.group1$age.group == "adult.31-40"],"adult.31-40",
                     ifelse(is.na(age.group) == TRUE &
                            CryoSleep == "False" &
                            luxury <age.group1$avg[
                              age.group1$age.group == "adult.31-40"] &
                            luxury >= age.group1$avg[
                              age.group1$age.group == "adult.51-60"],"adult.51-60",
                     ifelse(is.na(age.group) == TRUE &
                            CryoSleep == "False" &
                            luxury <age.group1$avg[
                              age.group1$age.group == "adult.51-60"] &
                            luxury >=age.group1$avg[
                              age.group1$age.group == "adult.41-50"],"adult.41-50",
                     ifelse(is.na(age.group) == TRUE &
                            CryoSleep == "False" &
                            luxury <age.group1$avg[
                              age.group1$age.group == "adult.41-50"] &
                            luxury >=age.group1$avg[
                              age.group1$age.group == "elder"],"elder",
                     ifelse(is.na(age.group) == TRUE &
                            CryoSleep == "False" &
                            luxury <age.group1$avg[
                              age.group1$age.group == "elder"] &
                            luxury >=age.group1$avg[
                              age.group1$age.group == "adult.21-30"],"adult.21-30",
                    ifelse(is.na(age.group) == TRUE &
                            CryoSleep == "False" &
                            luxury <age.group1$avg[
                              age.group1$age.group == "adult.21-30"] &
                            luxury >age.group1$avg[
                              age.group1$age.group == "child"],"teen",
                    ifelse(is.na(age.group) == TRUE &
                            CryoSleep == "False" &
                            luxury == age.group1$avg[
                              age.group1$age.group == "child"], "child",age.group))))))))

# let's compare the scores before and after treatment
age.group2 <- train.clean %>%
  filter(CryoSleep == "False") %>%
  select(luxury,
         age.group) %>%
  group_by(age.group) %>%
  summarise(n = n(),
            avg = mean(luxury)) %>%
  arrange(desc(avg))

data.frame(group = age.group1$age.group,
           Lux.avg.before = age.group1$avg,
           Lux.avg.after = age.group2$avg,
           n.before = age.group1$n,
           n.after = age.group2$n) %>%
  mutate(diff = n.after - n.before)

# making age.group and average age table 
mean.age1 <- train.clean %>% 
  filter(CryoSleep == "False") %>%
  select(Age,
         age.group) %>%
  group_by(age.group) %>%
  drop_na() %>%
  summarise(n = n(),
            avg = mean(Age),
            median = median(Age))

mean.luxury1 <- train.clean %>% 
  filter(CryoSleep == "False") %>%
  select(age.group,
         luxury) %>%
  group_by(age.group) %>%
  drop_na() %>%
  summarise(n = n(),
            avg = mean(luxury),
            median = median(luxury))

age.lux.1 <- data.frame(group = mean.luxury1$age.group,
           avg.luxury = mean.luxury1$avg,
           avg.age = mean.age1$avg,
           n = mean.age1$n) %>%
  arrange(desc(avg.luxury))

age.lux.1

# Fill Ages in for CryoSleep = False 
train.clean <- train.clean %>% 
  mutate(Age = ifelse(CryoSleep == "False" &
                        is.na(Age) == TRUE &
                        age.group == "adult.31-40",
                      age.lux.1$avg.age[age.lux.1$group == "adult.31-40"],
                      ifelse(CryoSleep == "False" &
                               is.na(Age) == TRUE &
                        age.group == "adult.51-60",
                        age.lux.1$avg.age[age.lux.1$group == "adult.51-60"],
                             ifelse(CryoSleep == "False" &
                                      is.na(Age) == TRUE &
                               age.group == "adult.41-50",
                               age.lux.1$avg.age[age.lux.1$group == "adult.41-50"],
                                    ifelse(CryoSleep == "False" &
                                             is.na(Age) == TRUE &
                                      age.group == "elder",
                                      age.lux.1$avg.age[age.lux.1$group == "elder"],
                                           ifelse(CryoSleep == "False" &
                                                    is.na(Age) == TRUE &
                                             age.group == "adult.21-30",
                                             age.lux.1$avg.age[age.lux.1$group == "adult.21-30"],
                                             ifelse(CryoSleep == "False" &
                                                      is.na(Age) == TRUE &
                                               age.group == "teen",
                                               age.lux.1$avg.age[age.lux.1$group == "teen"],
                                               ifelse(CryoSleep == "False" &
                                                        is.na(Age) == TRUE &
                                                        age.group == "child",
                                                      age.lux.1$avg.age[age.lux.1$group == "child"],Age))))))))

# making age.group and average age table 
mean.age2 <- train.clean %>% 
  filter(CryoSleep == "False") %>%
  select(Age,
         age.group) %>%
  group_by(age.group) %>%
  summarise(n = n(),
            avg = mean(Age),
            median = median(Age))

mean.age2

mean.luxury2 <- train.clean %>% 
  filter(CryoSleep == "False") %>%
  select(age.group,
         luxury) %>%
  group_by(age.group) %>%
  drop_na() %>%
  summarise(n = n(),
            avg = mean(luxury),
            median = median(luxury))

age.lux.2 <- data.frame(group = mean.luxury2$age.group,
                        avg.luxury = mean.luxury2$avg,
                        avg.age = mean.age2$avg,
                        n = mean.age2$n) %>%
  arrange(desc(avg.luxury))  

# Tables for ages 
data.frame(group = age.lux.2$group,
           avg.age.before = age.lux.1$avg.age,
           avg.age.after = age.lux.2$avg.age,
           n.before = age.lux.1$n,
           n.after = age.lux.2$n) %>%
  mutate(diff = n.after - n.before)

sum(is.na(train.clean$Age[train.clean$CryoSleep == "True"])) # we still have 82 NA values for ages.

AgeCryoSleep.yes.before <- train.clean %>%
  filter(CryoSleep == "True") %>%
  group_by(deck) %>%
  drop_na() %>%
  summarise(avg = mean(Age),
            n = n()) # for the remaining 82 NA values, we will use these 
                              # values for plotting in missing Age 

AgeCryoSleep.yes.before$avg[AgeCryoSleep.yes.before$deck == "A"]

train.clean <- train.clean %>% 
  mutate(Age = ifelse(CryoSleep == "True" &
                        is.na(Age) == TRUE &
                        deck == "A",
                      AgeCryoSleep.yes.before$avg[AgeCryoSleep.yes.before$deck == "A"],
                      ifelse(CryoSleep == "True" &
                               is.na(Age) == TRUE &
                               deck == "B",
                             AgeCryoSleep.yes.before$avg[AgeCryoSleep.yes.before$deck == "B"],
                             ifelse(CryoSleep == "True" &
                                      is.na(Age) == TRUE &
                                      deck == "C",
                                    AgeCryoSleep.yes.before$avg[AgeCryoSleep.yes.before$deck == "C"],
                                    ifelse(CryoSleep == "True" &
                                             is.na(Age) == TRUE &
                                             deck == "D",
                                           AgeCryoSleep.yes.before$avg[AgeCryoSleep.yes.before$deck == "D"],
                                           ifelse(CryoSleep == "True" &
                                                    is.na(Age) == TRUE &
                                                    deck == "E",
                                                  AgeCryoSleep.yes.before$avg[AgeCryoSleep.yes.before$deck == "E"],
                                                  ifelse(CryoSleep == "True" &
                                                           is.na(Age) == TRUE &
                                                           deck == "F",
                                                         AgeCryoSleep.yes.before$avg[AgeCryoSleep.yes.before$deck == "F"],
                                                         ifelse(CryoSleep == "True" &
                                                                  is.na(Age) == TRUE &
                                                                  deck == "G",
                                                                AgeCryoSleep.yes.before$avg[AgeCryoSleep.yes.before$deck == "G"],
                                                                ifelse(CryoSleep == "True" &
                                                                         is.na(Age) == TRUE &
                                                                         deck == "unknown",
                                                                       AgeCryoSleep.yes.before$avg[AgeCryoSleep.yes.before$deck == "unknown"],Age)))))))))

AgeCryoSleep.yes.after <- train.clean %>%
  filter(CryoSleep == "True") %>%
  group_by(deck) %>%
  summarise(avg = mean(Age),
            n = n())

data.frame(group = AgeCryoSleep.yes.before$deck,
           avg.before = AgeCryoSleep.yes.before$avg,
           avg.after = AgeCryoSleep.yes.after$avg,
           n.before = AgeCryoSleep.yes.before$n,
           n.after = AgeCryoSleep.yes.after$n) %>%
  mutate(diff = n.after - n.before)

which(colSums(is.na(train.clean))>0)

#-------------------------------unseen.clean-------------------------------------

#unseen.clean = Split Passenger ID & Cabin into separate columns
unseen.clean <- unseen %>%
  separate(PassengerId, c("party.identifiers","ps"),"_") %>%
  mutate(party.size = as.numeric(ps)) %>%
  select(!ps)

unseen.clean <- unseen.clean %>%
  separate(Cabin,c("deck","NUMBER","SIDE"),"/") %>%
  mutate(side = ifelse(SIDE == "P","Port","Starboard")) %>%
  select(!c(NUMBER,SIDE))

# Indicating if using VR deck
unseen.clean <- unseen.clean %>%
  mutate(VR.use = ifelse(VRDeck >0,"yes","no"))

which(colSums(is.na(unseen.clean))>0) # moneys, side, age, VR.use 

# Money related columns: assuming NA = 0 & create luxury col instead
RoomService = unseen.clean["RoomService"][is.na(unseen.clean["RoomService"])] <- 0
FoodCourt = unseen.clean["FoodCourt"][is.na(unseen.clean["FoodCourt"])] <- 0
ShoppingMall = unseen.clean["ShoppingMall"][is.na(unseen.clean["ShoppingMall"])] <- 0
Spa = unseen.clean["Spa"][is.na(unseen.clean["Spa"])] <- 0
VRDeck = unseen.clean["VRDeck"][is.na(unseen.clean["VRDeck"])] <- 0

unseen.clean <- unseen.clean %>%
  mutate(luxury = RoomService +FoodCourt +ShoppingMall +Spa +VRDeck)

# cryosleep 
unseen.clean %>%
  select(CryoSleep,
         luxury) %>%
  group_by(CryoSleep) %>%
  summarise(n = n(),
            avg = mean(luxury))

# people in cryosleep not spending money => luxury = 0 = true, luxury > 0 = false
unseen.clean <- unseen.clean %>%
  mutate(CryoSleep = ifelse(CryoSleep == "" & luxury ==0,"True",CryoSleep))

unseen.clean <- unseen.clean %>%
  mutate(CryoSleep = ifelse(CryoSleep == "" & luxury >0,"False",CryoSleep))

unseen.clean <- unseen.clean %>%
  mutate(VR.use = ifelse(CryoSleep == "True" &
                           VR.use == "unknown","no",VR.use))

# Putting "unknown" for categorical columns with " ":
unseen.clean <- unseen.clean %>%
  mutate(side = ifelse(is.na(side) ==TRUE,"unknown",side),
         VR.use = ifelse(is.na(VR.use) ==TRUE,"unknown", VR.use),
         HomePlanet = ifelse(HomePlanet == "","unknown",HomePlanet),
         Destination = ifelse(Destination == "","unknown",Destination),
         VIP = ifelse(VIP == "","unknown",VIP),
         deck = ifelse(deck == "","unknown",deck))

# Age: to make age easy to analyze, make age group instead 

unseen.clean <- unseen.clean %>%
  mutate(age.group = ifelse(Age >= 61, "elder",
                            ifelse(Age <=60 & Age >= 51, "adult.51-60",
                                   ifelse(Age <=50 & Age >= 41,"adult.41-50",
                                          ifelse(Age <=40 & Age >= 31,"adult.31-40",
                                                 ifelse(Age <=30 & Age >= 21,"adult.21-30",
                                                        ifelse(Age <=20 & Age >= 10,"teen","child")))))))

age.group3 <- unseen.clean %>%
  filter(CryoSleep == "False") %>%
  select(age.group,
         luxury) %>%
  group_by(age.group) %>%
  drop_na() %>%
  summarise(n = n(),
            avg = mean(luxury)) %>%
  arrange(desc(avg))

age.group3

unseen.clean <- unseen.clean %>%
  mutate(age.group = ifelse(is.na(age.group) == TRUE &
                              CryoSleep == "False" &
                              luxury >= age.group3$avg[
                                age.group3$age.group == "adult.31-40"],"adult.31-40",
                            ifelse(is.na(age.group) == TRUE &
                                     CryoSleep == "False" &
                                     luxury <age.group3$avg[
                                       age.group3$age.group == "adult.31-40"] &
                                     luxury >= age.group3$avg[
                                       age.group3$age.group == "adult.51-60"],"adult.51-60",
                                   ifelse(is.na(age.group) == TRUE &
                                            CryoSleep == "False" &
                                            luxury <age.group3$avg[
                                              age.group3$age.group == "adult.51-60"] &
                                            luxury >=age.group3$avg[
                                              age.group3$age.group == "adult.41-50"],"adult.41-50",
                                          ifelse(is.na(age.group) == TRUE &
                                                   CryoSleep == "False" &
                                                   luxury <age.group3$avg[
                                                     age.group3$age.group == "adult.41-50"] &
                                                   luxury >=age.group3$avg[
                                                     age.group3$age.group == "elder"],"elder",
                                                 ifelse(is.na(age.group) == TRUE &
                                                          CryoSleep == "False" &
                                                          luxury <age.group3$avg[
                                                            age.group3$age.group == "elder"] &
                                                          luxury >=age.group3$avg[
                                                            age.group3$age.group == "adult.21-30"],"adult.21-30",
                                                        ifelse(is.na(age.group) == TRUE &
                                                                 CryoSleep == "False" &
                                                                 luxury <age.group3$avg[
                                                                   age.group3$age.group == "adult.21-30"] &
                                                                 luxury >age.group3$avg[
                                                                   age.group3$age.group == "child"],"teen",
                                                               ifelse(is.na(age.group) == TRUE &
                                                                        CryoSleep == "False" &
                                                                        luxury == age.group3$avg[
                                                                          age.group3$age.group == "child"], "child",age.group))))))))

# let's compare the scores before and after treatment
age.group4 <- unseen.clean %>%
  filter(CryoSleep == "False") %>%
  select(luxury,
         age.group) %>%
  group_by(age.group) %>%
  summarise(n = n(),
            avg = mean(luxury)) %>%
  arrange(desc(avg))

age.group4

data.frame(group = age.group3$age.group,
           Lux.avg.before = age.group3$avg,
           Lux.avg.after = age.group4$avg,
           n.before = age.group3$n,
           n.after = age.group4$n) %>%
  mutate(diff = n.after - n.before)

# making age.group and average age table 
mean.age3 <- unseen.clean %>% 
  filter(CryoSleep == "False") %>%
  select(Age,
         age.group) %>%
  group_by(age.group) %>%
  drop_na() %>%
  summarise(n = n(),
            avg = mean(Age),
            median = median(Age))

mean.luxury3 <- unseen.clean %>% 
  filter(CryoSleep == "False") %>%
  select(age.group,
         luxury) %>%
  group_by(age.group) %>%
  drop_na() %>%
  summarise(n = n(),
            avg = mean(luxury),
            median = median(luxury))

age.lux.3 <- data.frame(group = mean.luxury3$age.group,
                        avg.luxury = mean.luxury3$avg,
                        avg.age = mean.age3$avg,
                        n = mean.age3$n) %>%
  arrange(desc(avg.luxury))

age.lux.3


# Fill Ages in for CryoSleep = False 
unseen.clean <- unseen.clean %>% 
  mutate(Age = ifelse(CryoSleep == "False" &
                        is.na(Age) == TRUE &
                        age.group == "adult.31-40",
                      age.lux.3$avg.age[age.lux.3$group == "adult.31-40"],
                      ifelse(CryoSleep == "False" &
                               is.na(Age) == TRUE &
                               age.group == "adult.51-60",
                             age.lux.3$avg.age[age.lux.3$group == "adult.51-60"],
                             ifelse(CryoSleep == "False" &
                                      is.na(Age) == TRUE &
                                      age.group == "adult.41-50",
                                    age.lux.3$avg.age[age.lux.3$group == "adult.41-50"],
                                    ifelse(CryoSleep == "False" &
                                             is.na(Age) == TRUE &
                                             age.group == "elder",
                                           age.lux.3$avg.age[age.lux.3$group == "elder"],
                                           ifelse(CryoSleep == "False" &
                                                    is.na(Age) == TRUE &
                                                    age.group == "adult.21-30",
                                                  age.lux.3$avg.age[age.lux.3$group == "adult.21-30"],
                                                  ifelse(CryoSleep == "False" &
                                                           is.na(Age) == TRUE &
                                                           age.group == "teen",
                                                         age.lux.3$avg.age[age.lux.3$group == "teen"],
                                                         ifelse(CryoSleep == "False" &
                                                                  is.na(Age) == TRUE &
                                                                  age.group == "child",
                                                                age.lux.3$avg.age[age.lux.3$group == "child"],Age))))))))

# making age.group and average age table 
mean.age4 <- unseen.clean %>% 
  filter(CryoSleep == "False") %>%
  select(Age,
         age.group) %>%
  group_by(age.group) %>%
  summarise(n = n(),
            avg = mean(Age),
            median = median(Age))

mean.luxury4 <- unseen.clean %>% 
  filter(CryoSleep == "False") %>%
  select(age.group,
         luxury) %>%
  group_by(age.group) %>%
  drop_na() %>%
  summarise(n = n(),
            avg = mean(luxury),
            median = median(luxury))

age.lux.4 <- data.frame(group = mean.luxury4$age.group,
                        avg.luxury = mean.luxury4$avg,
                        avg.age = mean.age4$avg,
                        n = mean.age4$n) %>%
  arrange(desc(avg.luxury))  

# Tables for ages 
data.frame(group = age.lux.4$group,
           avg.age.before = age.lux.3$avg.age,
           avg.age.after = age.lux.4$avg.age,
           n.before = age.lux.3$n,
           n.after = age.lux.4$n) %>%
  mutate(diff = n.after - n.before)

sum(is.na(unseen.clean$Age[unseen.clean$CryoSleep == "True"])) 

AgeCryoSleep.yes.before.1 <- unseen.clean %>%
  filter(CryoSleep == "True") %>%
  group_by(deck) %>%
  drop_na() %>%
  summarise(avg = mean(Age),
            n = n()) # for the remaining 34 NA values, we will use these 
# values for plotting in missing Age 

unseen.clean <- unseen.clean %>% 
  mutate(Age = ifelse(CryoSleep == "True" &
                        is.na(Age) == TRUE &
                        deck == "A",
                      AgeCryoSleep.yes.before.1$avg[AgeCryoSleep.yes.before.1$deck == "A"],
                      ifelse(CryoSleep == "True" &
                               is.na(Age) == TRUE &
                               deck == "B",
                             AgeCryoSleep.yes.before.1$avg[AgeCryoSleep.yes.before.1$deck == "B"],
                             ifelse(CryoSleep == "True" &
                                      is.na(Age) == TRUE &
                                      deck == "C",
                                    AgeCryoSleep.yes.before.1$avg[AgeCryoSleep.yes.before.1$deck == "C"],
                                    ifelse(CryoSleep == "True" &
                                             is.na(Age) == TRUE &
                                             deck == "D",
                                           AgeCryoSleep.yes.before.1$avg[AgeCryoSleep.yes.before.1$deck == "D"],
                                           ifelse(CryoSleep == "True" &
                                                    is.na(Age) == TRUE &
                                                    deck == "E",
                                                  AgeCryoSleep.yes.before.1$avg[AgeCryoSleep.yes.before.1$deck == "E"],
                                                  ifelse(CryoSleep == "True" &
                                                           is.na(Age) == TRUE &
                                                           deck == "F",
                                                         AgeCryoSleep.yes.before.1$avg[AgeCryoSleep.yes.before.1$deck == "F"],
                                                         ifelse(CryoSleep == "True" &
                                                                  is.na(Age) == TRUE &
                                                                  deck == "G",
                                                                AgeCryoSleep.yes.before.1$avg[AgeCryoSleep.yes.before.1$deck == "G"],
                                                                ifelse(CryoSleep == "True" &
                                                                         is.na(Age) == TRUE &
                                                                         deck == "unknown",
                                                                       AgeCryoSleep.yes.before.1$avg[AgeCryoSleep.yes.before.1$deck == "unknown"],Age)))))))))
AgeCryoSleep.yes.after.1 <- unseen.clean %>%
  filter(CryoSleep == "True") %>%
  group_by(deck) %>%
  summarise(avg = mean(Age),
            n = n())

data.frame(group = AgeCryoSleep.yes.before.1$deck,
           avg.before = AgeCryoSleep.yes.before.1$avg,
           avg.after = AgeCryoSleep.yes.after.1$avg,
           n.before = AgeCryoSleep.yes.before.1$n,
           n.after = AgeCryoSleep.yes.after.1$n) %>%
  mutate(diff = n.after - n.before)

which(colSums(is.na(unseen.clean))>0)

#----------------------Visualization,tests, & Picking Variables------------------

# Linear regression of Luxury, Party Size, and Age
train.clean.lm <- train.clean %>%
  mutate(Transported = ifelse(Transported == "True",1,0)) %>%
  select(Transported,
         luxury,
         Age,
         party.size) %>%
  data.frame()

summary(lm(data = train.clean.lm,
           Transported ~ luxury +Age +party.size))
# luxury, age, party size are significant predictors  

# HomePlanet as predictors
table(train.clean$Transported,train.clean$HomePlanet)
chisq.test(train.clean$Transported,train.clean$HomePlanet,correct = FALSE)

# Homeplanet is significant. Let's see the difference 
train.clean %>%
  group_by(HomePlanet,
           Transported) %>%
  summarise(n = n(),
            percent = round(n/nrow(train.clean)*100,2)) %>%
  data.frame() %>%
  ggplot(aes(x = HomePlanet,
             y = percent,
             fill = Transported)) +
  geom_bar(mapping = aes(x= HomePlanet,
                         y=percent),
           stat = "identity",
           width = 0.5, position = "dodge") +
  theme_minimal() +
  scale_fill_brewer(palette = "Paired") +
  xlab("Home Planet") +
  ylab("Total Passengers (%)") +
  ggtitle("Home Planet as Predictor for Transported Individuals")
# If you are from Earth, you are less likely to get transported; but if you are
#    from Europa, you are more likely to get transported. 

# CryoSleep as predictors
table(train.clean$Transported,train.clean$CryoSleep)
chisq.test(train.clean$Transported,train.clean$CryoSleep,correct = FALSE)
# CryoSleep is also significant. 
train.clean %>%
  group_by(Transported,
           CryoSleep) %>%
  summarise(n = n(),
            percent = n/nrow(train.clean)*100) %>%
  data.frame() %>%
  ggplot(aes(x = CryoSleep,
             y = percent,
             fill = Transported)) +
  geom_bar(mapping = aes(x =CryoSleep,
                         y =percent),
           stat = "identity",
           width = 0.5, position = "dodge") +
  scale_x_discrete(label = c("Awake","Sleep")) +
  theme_minimal() +
  scale_fill_brewer(palette = "Paired") +
  xlab("Cryo Sleep Status") +
  ylab("Total Passengers (%)") +
  ggtitle("Cryo Sleep Status as Predictor for Transported Individuals")
# If you awake, you are less likely to be transported than when you are put to sleep

# Destination
table(train.clean$Transported,train.clean$Destination)
chisq.test(train.clean$Transported,train.clean$Destination,correct = FALSE)

train.clean %>%
  group_by(Transported,
           Destination) %>%
  summarise(n = n(),
            percent = n/nrow(train.clean)*100) %>%
  data.frame() %>%
  ggplot(aes(x = Destination,
             y = percent,
             fill = Transported)) +
  geom_bar(mapping = aes(x =Destination,
                         y =percent),stat = "identity",
           width = 0.5, position = "dodge") +
  theme_minimal() +
  scale_fill_brewer(palette = "Paired") +
  xlab("Destination") +
  ylab("Total Passengers (%)") +
  ggtitle("Destination as Predictor for Transported Individuals")
# those who arrive at 55 Cancrie are more likely to transport, but TRAP are not

# deck as predictors
table(train.clean$Transported,train.clean$deck)
chisq.test(train.clean$Transported,train.clean$deck,correct = FALSE)

train.clean %>%
  group_by(Transported,
           deck) %>%
  summarise(n = n(),
            percent = n/nrow(train.clean)*100) %>%
  data.frame() %>%
  ggplot(aes(x = deck,
             y = percent,
             fill = Transported)) +
  geom_bar(mapping = aes(x= deck,
                         y= percent),
           stat = "identity",
           width = 0.5, position = "dodge") +
  scale_x_discrete(label = c("A","B","C","D","E","F","G","T","unknown")) +
  theme_minimal()+
  scale_fill_brewer(palette = "Paired") +
  xlab("Cabin Position (Deck)") +
  ylab("Total Passengers (%)") +
  ggtitle("Spaceship Deck Position as Predictor for Transported Individuals")
# Transported: B, C, & G
# Not Transported: D,E, & F

# side as predictors
table(train.clean$Transported,train.clean$side)
chisq.test(train.clean$Transported,train.clean$side,correct = FALSE)

train.clean %>%
  group_by(Transported,
           side) %>%
  summarise(n = n(),
            percent = n/nrow(train.clean)*100) %>%
  data.frame() %>%
  ggplot(aes(x = side,
             y = percent,
             fill = Transported)) +
  geom_bar(mapping = aes(x= side,
                         y= percent),
           stat = "identity",
           width = 0.5,
           position = "dodge") +
  scale_x_discrete(label = c("Port","Starboard","unknown")) +
  theme_minimal() +
  scale_fill_brewer(palette = "Paired") +
  xlab("Cabin Position (Side)") +
  ylab("Total Passengers (%)") +
  ggtitle("Cabin Position (Side) as Predictor for Transported Individuals")
# TRansported: Starboard
# Not Transported: Port

# VIP as predictors
table(train.clean$Transported,train.clean$VIP)
chisq.test(train.clean$Transported,train.clean$VIP,correct = FALSE)

train.clean %>%
  group_by(Transported,
           VIP) %>%
  summarise(n = n(),
            percent = n/nrow(train.clean)*100) %>%
  data.frame() %>%
  ggplot(aes(x = VIP,
             y = n,
             fill = Transported)) +
  geom_bar(mapping = aes(x= VIP,
                         y= percent),
           stat = "identity",
           width = 0.5, position = "dodge") +
  scale_x_discrete(label = c("False","True","Unknown",)) +
  theme_minimal() +
  scale_fill_brewer(palette = "Paired") +
  xlab("VIP Status") +
  ylab("Total Passengers (%)") +
  ggtitle("VIP Status as Predictor for Transported Individuals")
# it is not that big of a difference. discard this. 


# VR.use as predictor
table(train.clean$Transported,train.clean$VR.use)
chisq.test(train.clean$Transported,train.clean$VR.use,correct = FALSE)

train.clean %>%
  group_by(Transported,
           VR.use) %>%
  summarise(n = n(),
            percent = n/nrow(train.clean)*100) %>%
  data.frame() %>%
  ggplot(aes(x = VR.use,
             y = percent,
             fill = Transported)) +
  geom_bar(mapping = aes(x= VR.use,
                         y= percent),
           stat = "identity",
           width = 0.5,
           position = "dodge") +
  scale_x_discrete(label = c("No","unknown","Yes")) +
  theme_minimal() +
  scale_fill_brewer(palette = "Paired") +
  xlab("Using VR Deck") +
  ylab("Total Passengers (%)") +
  ggtitle("Using VR (yes vs. no) as Predictor for Transported Individuals")
# TRansported: Not using VR Deck
# Not Transported: using VR deck

#-------------------------------------Predictor---------------------------------

# Continuous predictors: luxury, age, party size
# categorical predictors: 
        # Homeplanet: Earth vs. Europa
        # CryoSleep: 
        # Destination: TRAPPIST or 55
        # Deck: B, C, D, E, F, G only 
        # side: Starboard vs. Port
        # VR.use: use vs. not use 

# Discard predictors: VIP status 

# ------------------Dummy Coding -----------------------------------------------
# Train.clean
train.dummy <- train.clean %>%
  mutate(Transported = ifelse(Transported == "True",1,0),
         CryoSleep = ifelse(CryoSleep == "True",1,0),
         side.starboard = ifelse(side == "Starboard",1,0),
         side.port = ifelse(side == "Port",1,0),
         Earth.home = ifelse(HomePlanet == "Earth",1,0),
         Europa.home = ifelse(HomePlanet == "Europa",1,0),
         dest.TRAP = ifelse(Destination == "TRAPPIST-1e",1,0),
         dest.55 = ifelse(Destination == "55 Cancri e",1,0),
         deck.B = ifelse(deck == "B",1,0),
         deck.C = ifelse(deck == "C",1,0),
         deck.D = ifelse(deck == "D",1,0),
         deck.E = ifelse(deck == "E",1,0),
         deck.F = ifelse(deck == "F",1,0),
         deck.G = ifelse(deck == "G",1,0),
         VR.use = ifelse(VR.use == "yes",1,0))

# Unseen.Clean
unseen.dummy <- unseen.clean %>%
  mutate(CryoSleep = ifelse(CryoSleep == "True",1,0),
         side.starboard = ifelse(side == "Starboard",1,0),
         side.port = ifelse(side == "Port",1,0),
         Earth.home = ifelse(HomePlanet == "Earth",1,0),
         Europa.home = ifelse(HomePlanet == "Europa",1,0),
         dest.TRAP = ifelse(Destination == "TRAPPIST-1e",1,0),
         dest.55 = ifelse(Destination == "55 Cancri e",1,0),
         deck.B = ifelse(deck == "B",1,0),
         deck.C = ifelse(deck == "C",1,0),
         deck.D = ifelse(deck == "D",1,0),
         deck.E = ifelse(deck == "E",1,0),
         deck.F = ifelse(deck == "F",1,0),
         deck.G = ifelse(deck == "G",1,0),
         VR.use = ifelse(VR.use == "yes",1,0))

#-------------------Slicing index for Train set----------------------------------

train.dummy.1 <- train.dummy %>%
  select(!c(party.identifiers,
            HomePlanet,
            deck,
            VIP,
            RoomService,
            FoodCourt,
            ShoppingMall,
            Spa,
            VRDeck,
            Name,
            side,
            Destination,
            age.group)) %>%
  mutate(CryoSleep = factor(CryoSleep),
         side.starboard = factor(side.starboard),
         side.port = factor(side.port),
         Earth.home = factor(Earth.home),
         Europa.home = factor(Europa.home),
         dest.TRAP = factor(dest.TRAP),
         dest.55 = factor(dest.55),
         deck.B = factor(deck.B),
         deck.C = factor(deck.C),
         deck.D = factor(deck.D),
         deck.E = factor(deck.E),
         deck.F = factor(deck.F),
         VR.use = factor(VR.use),
         Transported = factor(Transported))
      
# factoring Unseen data
unseen.dummy.1 <- unseen.dummy %>%
  select(!c(party.identifiers,
            HomePlanet,
            deck,
            VIP,
            RoomService,
            FoodCourt,
            ShoppingMall,
            Spa,
            VRDeck,
            Name,
            side,
            Destination,
            age.group)) %>%
  mutate(CryoSleep = factor(CryoSleep),
         side.starboard = factor(side.starboard),
         side.port = factor(side.port),
         Earth.home = factor(Earth.home),
         Europa.home = factor(Europa.home),
         dest.TRAP = factor(dest.TRAP),
         dest.55 = factor(dest.55),
         deck.B = factor(deck.B),
         deck.C = factor(deck.C),
         deck.D = factor(deck.D),
         deck.E = factor(deck.E),
         deck.F = factor(deck.F),
         VR.use = factor(VR.use))

train.index <- createDataPartition(train.dummy.1$Transported,
                                   p = 0.85,
                                   times = 1,
                                   list = FALSE) 

train.set <- train.dummy.1 %>% slice(train.index)  
test.set <- train.dummy.1 %>% slice(-train.index) 
# total number of predictors = 17

#---------------------------glm model------------------------------------------

train_glm <- train(Transported ~.,
                   method = "glm",
                   data = train.set)

y_hat_glm <- predict(train_glm,
                     test.set,
                     type = "raw")

cmglm <- confusionMatrix(y_hat_glm,
                         test.set$Transported)

data.frame(Accuracy = cmglm$overall["Accuracy"],
           Sensitivity = cmglm$byClass["Sensitivity"],
           Specificity = cmglm$byClass["Specificity"])


#---------------------------knn method------------------------------------------

control <- trainControl(method = "cv",
                        number = 10,
                        p = 0.9)

train_knn <- train(Transported ~.,
                   method = "knn",
                   data = train.set,
                   tuneGrid = data.frame(k = c(3:18)),
                   trControl = control)

fit_knn <- knn3(Transported ~ .,
                data = train.set,
                k = train_knn$bestTune$k)

y_hat_knn <- predict(fit_knn,
                     test.set,
                     type = "class")

cmknn <- confusionMatrix(y_hat_knn,
                         test.set$Transported)

data.frame(Accuracy = cmknn$overall["Accuracy"],
           Sensitivity = cmknn$byClass["Sensitivity"],
           Specificity = cmknn$byClass["Specificity"])

#----------------------------Classification tree--------------------------------

# train algorithm
train_rpart <- train(Transported ~ .,
                     method = "rpart",
                     data = train.set,
                     tuneGrid = data.frame(cp = seq(0,2, len = 30)))

y_hat_rpart <- predict(train_rpart,
                       test.set)

cmrpart <- confusionMatrix(y_hat_rpart,
                           test.set$Transported)

data.frame(Accuracy = cmrpart$overall["Accuracy"],
           Sensitivity = cmrpart$byClass["Sensitivity"],
           Specificity = cmrpart$byClass["Specificity"])


#----------------------------Random Forest ----------------------------------
# this code takes long
train_rf <- train(Transported ~.,
                  method = "rf",
                  data = train.set,
                  tuneGrid = data.frame(mtry = seq(1:18)),
                  ntree = 500)

fit_rf <- randomForest(Transported ~.,
                       data = train.set,
                       minNode = 4)

y_hat_rf <- predict(fit_rf,
                    test.set)

cmrf <- confusionMatrix(y_hat_rf,
                        test.set$Transported)

data.frame(Accuracy = cmrf$overall["Accuracy"],
           Sensitivity = cmrf$byClass["Sensitivity"],
           Specificity = cmrf$byClass["Specificity"])

#------------------------Comparison Table-------------------------------------

Method <- c("Generalized Linear Model",
            "K-Nearest-Neighbor",
            "Classification Tree",
            "Random Forest")

Accuracy <- c(cmglm$overall["Accuracy"],
              cmknn$overall["Accuracy"],
              cmrpart$overall["Accuracy"],
              cmrf$overall["Accuracy"])

Sensitivity <- c(cmglm$byClass["Sensitivity"],
                 cmknn$byClass["Sensitivity"],
                 cmrpart$byClass["Sensitivity"],
                 cmrf$byClass["Sensitivity"])

Specificity <- c(cmglm$byClass["Specificity"],
                 cmknn$byClass["Specificity"],
                 cmrpart$byClass["Specificity"],
                 cmrf$byClass["Specificity"])

data.frame(Method,Accuracy,Sensitivity,Specificity)

# ------------------Predict the Unseen-------------------------------------
predict_unseen <- predict(fit_rf,
                          newdata = unseen.dummy.1,
                          type = "prob")

predict_unseen <- as.data.frame(predict_unseen) 
colnames(predict_unseen) <- c("predict.no","predict.yes")

predict_unseen <- predict_unseen %>%
  mutate(Transported = ifelse(predict.yes > predict.no, "True","False"))

predict_unseen <- cbind(unseen$PassengerId,predict_unseen)

predict_unseen <- predict_unseen %>%
  select(!c(predict.no,
            predict.yes))
colnames(predict_unseen) <- c("PassengerId","Transported")

# save to csv 
write.csv(predict_unseen,
          "C:\\Users\\minhk\\OneDrive\\Data Science stuffs\\PORTFOLIO\\ML - Titanic Spaceship\\predict3.csv",
          row.names = FALSE)
