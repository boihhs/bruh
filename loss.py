import torch
import torch.nn.functional as F

class customLoss(torch.nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.pos_neg_ratio = 3

    def forward(self, predictions, labels, device):

        '''
        Predictions is the output of the model (batch size, 150, 3)
        labels are the ground truth in global coords as percentage (batch size, num of weeds, 2)
        '''

        # Generate Default Points
        numberOfPointsAlongHorizontal = 40
        default_points_x = torch.linspace(1/(2*numberOfPointsAlongHorizontal), 1-1/(2*numberOfPointsAlongHorizontal), numberOfPointsAlongHorizontal)
        
        numberOfPointsAlongVertical = 40
        default_points_y = torch.linspace(1/(2*numberOfPointsAlongVertical), 1-1/(2*numberOfPointsAlongVertical), numberOfPointsAlongVertical)
        
        default_points = torch.cartesian_prod(default_points_x, default_points_y).to(device)


        # Calculation Loss
        loss_total = 0
        
        '''
        Confidence Loss: in pos set do -log(c) in negative set do -log(1-c)
        '''
        # Get the confidence scores
        conf = torch.exp(predictions[:, 0])/(torch.exp(predictions[:, 0])+torch.exp(torch.ones(1,numberOfPointsAlongHorizontal*numberOfPointsAlongVertical).to(device)-predictions[:, 0]))
        # Get the -log values
        pos_confidence_loss_log = -torch.log(conf + 1e-8)
        neg_confidence_loss_log = -torch.log(torch.ones(1,numberOfPointsAlongHorizontal*numberOfPointsAlongVertical).to(device)-conf + 1e-8)

        for j in range(labels.shape[0]):
            # Get Ground Truth
            ground_truth = labels[j, :].repeat(numberOfPointsAlongHorizontal*numberOfPointsAlongVertical, 1)

            '''
            Matching (gives us a 150x1 matching vector for positive and negative sets)
            '''
            # Get distance from all default points to the label
            distance = torch.abs(default_points[:, 0] - ground_truth[:, 0]) + torch.abs(default_points[:, 1] - ground_truth[:, 1])
            
            # Get Positive Set
            min_distance_for_positive_set = .025
            pos_mask = (distance <= min_distance_for_positive_set).float()
            # Get Negative Set
            num_pos = pos_mask.sum().int().item()
     
            num_neg = numberOfPointsAlongHorizontal*numberOfPointsAlongVertical-num_pos
            neg_distances = (distance*(distance > min_distance_for_positive_set).float())
            neg_distances[neg_distances == 0] = 99 # Replaces all zeros with a 99
            _, neg_set_indexes = torch.topk(neg_distances, k=num_neg, largest=False)

            neg_mask = torch.zeros(1, numberOfPointsAlongHorizontal*numberOfPointsAlongVertical).to(device)
            neg_mask[0, neg_set_indexes] = 1

            loss_conf = (pos_confidence_loss_log*pos_mask).sum()/num_pos+3*(neg_confidence_loss_log*neg_mask).sum()/num_neg

            loss_total = loss_total + (loss_conf)/ labels.shape[0]

        return loss_total