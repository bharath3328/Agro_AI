def assess_disease_intelligence(confidence, cam_coverage):
    if confidence > 0.9 and cam_coverage > 0.6:
        stage = "Late"
        action = "Immediate Treatment"
        yield_loss = "35–60%"
    elif confidence > 0.75:
        stage = "Mid"
        action = "Curative Treatment"
        yield_loss = "15–30%"
    else:
        stage = "Early"
        action = "Preventive Monitoring"
        yield_loss = "5–10%"

    return stage, action, yield_loss
