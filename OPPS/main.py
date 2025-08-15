# # This is for the normal values
# class Item:
#         def __init__(self, name, price, quantity=10): ## it is by default parameter
#                 self.name = name
#                 self.price = price
#                 self.quantity = quantity

#         def calculate(self):
#                 return self.price*self.quantity




# item1 = Item('Phone', 100, 33)
# item2 = Item('Laptop', 1000)

# # print(item1.name)
# # print(item2.price)

# print(item1.calculate())


# This class will be validate all the input data
class Item:
        pay_rate = 0.8
        def __init__(self, name: str, price: float, quantity=10): ## This is taking the name and price parameter as typed in his data typed

                # Run validations to the receiving arguments
                assert price >=0, f"Price should not be negative!"
                assert quantity >=0, f"Quantity should not be negative!"

                '''This assert set the limit of your variable and it is also show you the your custom error massage'''




                # Assign to self object
                self.name = name
                self.price = price
                self.quantity = quantity

        def calculate(self):
                return self.price*self.quantity
        
        def apply_discount(self):
               self.price = self.price*self.pay_rate
               return self.price




item1 = Item('Phone', 100, 33)
item2 = Item('Laptop', 1000)

''' If we create a Class level instance then that instance is accessible from both the class as well as their instances'''
print(item1.pay_rate)
print(Item.pay_rate)

# print(Item.__dict__) ## it is showing the all attributes present in the class
# item2.pay_rate = 0.7
item2.apply_discount()
print(item2.price)