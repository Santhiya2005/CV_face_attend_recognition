const mongoose = require('mongoose');

const employeeSchema = new mongoose.Schema({
    name: { type: String, required: true },
    
    position: { type: String, required: true },
    department: { type: String, required: true },
    salary: { type: Number, required: true },
    dob: { type: Date, required: true },
    dateOfJoining: { type: Date, required: true },
    maritalStatus: { type: String, required: true },
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true },
    phoneNumber: { type: String, required: true },
    address: { type: String, required: true },
    emergencyContact: { type: String, required: true },
    relationshipToEmergency: { type: String, required: true },
    bloodGroup: { type: String, required: true },
    education: { type: String },
    languagesKnown: [{ type: String }],
    aadharNo: { type: String, required: true },
    panNo: { type: String, required: true },
    githubId: { type: String },
    linkedIn: { type: String },
});

const Employee = mongoose.model('Employee', employeeSchema);
module.exports = Employee;
