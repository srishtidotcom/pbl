const dotenv = require('dotenv')
dotenv.config()

const express = require('express')
const cors = require('cors')
const connectCloudinary = require('./config/cloudinary')
const connectDatabase = require('./config/mongodb')

const userRouter = require('./routes/user_routes')
const financialProfileRouter = require('./routes/financialProfile_routes')
const loanEligibilityRouter = require('./routes/loanEligibilityCheck_routes')
const adminRouter = require('./routes/admin_routes')
const loanProductRouter = require('./routes/loanProduct_routes')

const { verifyLoanOnChain } = require('./services/blockchainservice');
// app config 
const app = express()
const port = process.env.PORT
connectDatabase()

// connectCloudinary()

// middlewares 
// middlewares 
app.use(cors({
    origin: process.env.FRONTEND_URL,
    credentials: true
}))
app.use(express.json())
app.use(express.json())

// routes
app.use('/user', userRouter)
app.use('/financial-profile', financialProfileRouter)
app.use('/loan-eligibility', loanEligibilityRouter)
app.use('/api/admin', adminRouter)
app.use('/api/loan-products', loanProductRouter)

app.get('/api/verify-loan/:id', async (req, res) => {
    try {
        const result = await verifyLoanOnChain(req.params.id);
        res.json(result);
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'ok' });
});
// run server
app.listen(port, () => {
    console.log('App running on port ' + port)
})