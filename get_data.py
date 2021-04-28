

from kinesis.producer import KinesisProducer


k = KinesisProducer(stream_name="arn:aws:kinesis:us-east-1:384393252157:stream/get_users")


from randomuser import RandomUser
import json
# Generate a single user
user = RandomUser()

# Generate a list of 10 random users
user_list = RandomUser.generate_users(10)
for user in user_list:
    x={"name":user.get_full_name(),
    "age":user.get_age(),
    "gender":user.get_gender(),
    "zip":user.get_zipcode()}
    record=json.dumps(x)
    k.put(record)
