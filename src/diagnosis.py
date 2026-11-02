def diagnose_and_recover(predicted_label, node_id):
    """
    Map the detector's prediction to a diagnosis and recovery action.

    This is a rule-based prototype policy, not a deployed
    wireless routing protocol.
    """

    if predicted_label == "normal":
        diagnosis = "Normal operation"
        action = "Continue routing"
        node_status = "active"

    elif predicted_label == "fault":
        diagnosis = "Likely sensor or communication fault"
        action = "Mark node unreliable and reroute"
        node_status = "unreliable"

    elif predicted_label == "blackhole":
        diagnosis = "Likely blackhole attack"
        action = "Isolate node and reroute"
        node_status = "isolated"

    else:
        diagnosis = "Unknown behavior"
        action = "Flag for further inspection"
        node_status = "under_review"

    return {
        "node_id": node_id,
        "predicted_label": predicted_label,
        "diagnosis": diagnosis,
        "recovery_action": action,
        "node_status": node_status,
    }


if __name__ == "__main__":
    test_cases = [
        (0, "normal"),
        (1, "fault"),
        (2, "blackhole"),
    ]

    print("\nDiagnosis and Recovery Test\n")

    for node_id, prediction in test_cases:
        result = diagnose_and_recover(prediction, node_id)

        print(f"Node {result['node_id']}:")
        print(f"  Prediction: {result['predicted_label']}")
        print(f"  Diagnosis: {result['diagnosis']}")
        print(f"  Recovery: {result['recovery_action']}")
        print(f"  Status: {result['node_status']}")
        print()