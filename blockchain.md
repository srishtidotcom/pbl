## ⛓️ Blockchain Audit & Data Integrity (Live)

Every loan eligibility check and its corresponding SHAP explanation data (`summary`, `topPositive`, `topNegative`, `baseValue`) is now anchored to the **Ethereum Sepolia Testnet** via **IPFS**. This ensures that the decision logic is immutable and verifiable.

**The Blockchain Integration provides:**
- **Decentralized Storage:** Full result metadata is pinned to IPFS via Pinata.
- **On-Chain Anchoring:** The IPFS CID is stored in a Solidity smart contract for permanent auditing.
- **Tamper-Proof Verification:** Users receive a unique Transaction Hash per eligibility check to verify their results on Etherscan.

The `mlExplanation` object used for this decentralized anchor looks like:

```json
{
  "summary": "Your application looks strong. Key strengths: Credit Score and Monthly Income.",
  "topPositive": [{ "feature": "credit_score", "label": "Credit Score", "shap_value": 0.42, "direction": "positive", "magnitude": "high" }],
  "topNegative": [{ "feature": "dti_ratio", "label": "Debt-to-Income Ratio", "shap_value": -0.18, "direction": "negative", "magnitude": "medium" }],
  "baseValue": 0.48,
  "blockchainTxHash": "0x621b557a5cd8d839e98dd0062ce1cb172de4bd91f1b84699eb56945ba1088349",
  "ipfsHash": "QmfQh3D4wA5GQwm6bBstHXD2Ygi97C5WQ7yQ2a6Psj5wLB"
}
