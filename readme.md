# Product recommender

This project is an attempt on creating a simple recommender system to manage the purchase of a limited number of products offered by one of our clients to their clients.

## Approach
There are multiple paradigms in recommender systems, so I'll exploit the 2 main ones and try to create a hybrid system to work around their flaws.

The ones I'll be using are content-based and collaborative-based recommender systems.

Content-based approaches have the advantage of using the user's data to figure out their preferences for products in a specific domain. This is useful when the user has given limited feedback.

Collaborative approaches, on the other end, rely heavily on user feedback to compare users' preferences, so it's useful when evaluating existing users to find new products to recommend.

Thus, the system will divide the use of those paradigms into two different regions of the user lifecycle.

The first one is the user registration and first contact with the products. The second is the suggestion of new products to existing users

### First contact
When a new user is signed up, they need to provide some basic information, like the scope of their company, the size, number of employees, location, etc. This data is used to create the profile of the user, that is used to generate predictions about the likelihood of the user purchasing a given product.

Therefore, I model the interaction of every user in the database with every product and separate the models into domains. This 'transforms' the problem into one of supervised learning since I have discrete definitions of every user's interaction with every product.

Also, since the number of products is very limited (about 22), I'm able to model each interaction separately and run per-product predictions.

I'm using a simple bayesian approach to start, so that the result of the prediction is a continuous probability that I can rank from more likely to less likely.

You can find a more in-depth analysis of this model at `results/registration/`

### Collaborative recommendation
Now imagine that we have lots of already-registered users that are using our customer's products. We want them to purchase more, so we need to find out what to recommend them to buy.

To do this, we need a collaborative model that learns how each product relates to one another and how they translate to trends in user activity.
