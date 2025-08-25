
library(marginaleffects)
mod <- glm(am ~ hp * wt + factor(cyl), data = mtcars, family = binomial)
# vcov(mod)
p <- predictions(mod, type = "response")

components(p, "jacobian") |> head()
