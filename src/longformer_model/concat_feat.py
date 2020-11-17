# result = { "negative": { "confidence": 1.9531265169053625e-8 }, "neutral": { "confidence": 0.0019531445243914027 }, "positive": { "confidence": 0.9960937499035479 } }



# def flatten_dict(d):
#     def expand(key, value):
#         if isinstance(value, dict):
#             return [ (key, v) for k, v in flatten_dict(value).items() ]
#         else:
#             return [ (key, value) ]

#     items = [ item for k, v in d.items() for item in expand(k, v) ]

#     return dict(items)

# d = flatten_dict(result)

# d = max(d, key=d.get)

# print(d)
