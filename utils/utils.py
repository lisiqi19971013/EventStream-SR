import numpy as np
import torch


def getNeuronConfig(type: str='SRMALPHA',
                    theta: float=10.,
                    tauSr: float=1.,
                    tauRef: float=1.,
                    scaleRef: float=2.,
                    tauRho: float=0.3,  # Was set to 0.2 previously (e.g. for fullRes run)
                    scaleRho: float=1.):
    """
    :param type:     neuron type
    :param theta:    neuron threshold
    :param tauSr:    neuron time constant
    :param tauRef:   neuron refractory time constant
    :param scaleRef: neuron refractory response scaling (relative to theta)
    :param tauRho:   spike function derivative time constant (relative to theta)
    :param scaleRho: spike function derivative scale factor
    :return: dictionary
    """
    return {
        'type': type,
        'theta': theta,
        'tauSr': tauSr,
        'tauRef': tauRef,
        'scaleRef': scaleRef,
        'tauRho': tauRho,
        'scaleRho': scaleRho,
    }


def getNeuronConfig1(type: str='LOIHI',
                    vThMant: int=80,
                    vDecay: int=128,
                    iDecay: int=1024,
                    refDelay: int=1,
                    wgtExp: int=0,
                    tauRho: int=1,
                    scaleRho: int=1):

    return {
        "type": type,
        "vThMant": vThMant,
        "vDecay": vDecay,
        "iDecay": iDecay,
        "refDelay": refDelay,
        "wgtExp": wgtExp,
        "tauRho": tauRho,
        "scaleRho": scaleRho
    }



# def getEventFromTensor(eventTensor):
#     bs = eventTensor.shape[0]
#     eventList = []
#     for b in range(bs):
#         e = eventTensor[b]
#         event = []
#         for k in torch.nonzero(e):
#             event.append(np.array([k[3].item(), k[1].item(), k[2].item(), k[0].item()]))
#         event = np.vstack(event)
#         event = event[event[:,0].argsort()]
#         eventList.append(event)
#     return eventList


def getEventFromTensor(eventTensor):
    if len(eventTensor.shape) == 5:
        bs = eventTensor.shape[0]
        eventList = []
        for b in range(bs):
            e = eventTensor[b]
            event = torch.nonzero(e).cpu().numpy()
            event = event[:,[3,1,2,0]]
            event = event[event[:,0].argsort()]
            eventList.append(event)
    elif len(eventTensor.shape) == 4:
        event = torch.nonzero(eventTensor).cpu().numpy()
        event = event[:, [3, 1, 2, 0]]
        eventList = event[event[:, 0].argsort()]
    else:
        raise ValueError

    return eventList

