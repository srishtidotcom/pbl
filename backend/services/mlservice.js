const axios = require("axios");

const ML_URL   = process.env.ML_SERVICE_URL   || "http://localhost:8000";
const SHAP_URL = process.env.SHAP_SERVICE_URL || "http://localhost:8001";

/**
 * Builds the payload the Python pipeline expects from a
 * FinancialProfile doc + LoanEligibilityCheck request body
 */
function buildMLPayload(profile, requestedLoanAmount, tenureMonths) {
    const totalExistingEmi =
        profile.existingEmis.reduce((s, e) => s + e.monthlyAmount, 0) +
        profile.creditCardDues.reduce((s, c) => s + c.minimumDue, 0) +
        profile.otherLoans.reduce((s, l) => s + l.monthlyEMI, 0);

    return {
        age:                      profile.age,
        employment_type:          profile.employmentType,
        city_tier:                profile.cityTier,
        has_coapplicant:          0,                        // updated below if needed
        monthly_income:           profile.monthlyNetIncome,
        credit_score:             profile.creditScore,
        total_existing_emi:       totalExistingEmi,
        requested_loan_amount:    requestedLoanAmount,
        loan_tenure_months:       tenureMonths,
        work_experience_years:    profile.employmentTenureMonths / 12,
    };
}

async function getMLPrediction(payload) {
    const { data } = await axios.post(`${ML_URL}/predict`, payload, { timeout: 10000 });
    return data;
}

async function getMLExplanation(payload) {
    const { data } = await axios.post(`${SHAP_URL}/explain`, payload, { timeout: 15000 });
    return data;
}

async function getFullMLResult(profile, requestedLoanAmount, tenureMonths, hasCoApplicant = false) {
    const payload = buildMLPayload(profile, requestedLoanAmount, tenureMonths);
    payload.has_coapplicant = hasCoApplicant ? 1 : 0;

    const [prediction, explanation] = await Promise.all([
        getMLPrediction(payload),
        getMLExplanation(payload),
    ]);

    // Map snake_case SHAP response → camelCase for frontend
    const mappedExplanation = {
        verdict:          explanation.verdict,
        summary:          explanation.summary,
        topPositive:      explanation.top_positive,
        topNegative:      explanation.top_negative,
        allContributions: explanation.all_contributions,
        baseValue:        explanation.base_value,
    };

    return { prediction, explanation: mappedExplanation };
}

module.exports = { getFullMLResult };
