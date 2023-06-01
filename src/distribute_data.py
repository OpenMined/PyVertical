class Distribute_MNIST:
    """
  This class distribute each image among different workers
  It returns a dictionary with key as data owner's id and 
  value as a pointer to the list of data batches at owner's 
  location.
  
  example:-  
  >>> from distribute_data import Distribute_MNIST
  >>> obj = Distribute_MNIST(data_owners= (alice, bob, claire), data_loader= torch.utils.data.DataLoader(trainset)) 
  >>> obj.data_pointer[1]['alice'].shape, obj.data_pointer[1]['bob'].shape, obj.data_pointer[1]['claire'].shape
   (torch.Size([64, 1, 9, 28]),
    torch.Size([64, 1, 9, 28]),
    torch.Size([64, 1, 10, 28]))
  """

    def __init__(self, data_owners, data_loader):

        """
         Args:
          data_owners: tuple of data owners
          data_loader: torch.utils.data.DataLoader for MNIST 

        """

        self.data_owners = data_owners
        self.data_loader = data_loader
        self.no_of_owner = len(data_owners)

        self.data_pointer = []
        """
        self.data_pointer:  list of dictionaries where 
        (key, value) = (id of the data holder, a pointer to the list of batches at that data holder).
        example:
        self.data_pointer  = [
                                {"alice": pointer_to_alice_batch1, "bob": pointer_to_bob_batch1},
                                {"alice": pointer_to_alice_batch2, "bob": pointer_to_bob_batch2},
                                ...
                             ]
        """

        self.labels = []

        # iterate over each batch of dataloader for, 1) spliting image 2) sending to VirtualWorker
        for images, labels in self.data_loader:

            curr_data_dict = {}

            # calculate width and height according to the no. of workers for UNIFORM distribution
            height = images.shape[-1]//self.no_of_owner

            self.labels.append(labels)

            # iterate over each worker for distribution of current batch of the self.data_loader
            for i, owner in enumerate(self.data_owners[:-1]):

                # split the image and send it to VirtualWorker (which is supposed to be a dataowner or client)
                image_part_ptr = images[:, :, :, height * i : height * (i + 1)].send(
                    owner
                )

                curr_data_dict[owner.id] = image_part_ptr

            # Repeat same for the remaining part of the image
            last_owner = self.data_owners[-1]
            last_part_ptr = images[:, :, :, height * (i + 1) :].send(last_owner)

            curr_data_dict[last_owner.id] = last_part_ptr

            self.data_pointer.append(curr_data_dict)
            
    def __iter__(self):
        
        for data_ptr, label in zip(self.data_pointer[:-1], self.labels[:-1]):
            yield (data_ptr, label)
            
    def __len__(self):
        
        return len(self.data_loader)-1
            
