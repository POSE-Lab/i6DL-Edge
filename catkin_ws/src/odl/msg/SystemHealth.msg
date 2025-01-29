# unique string id for the id; format <organizationName>.<moduleName>.<submoduleName>.<type>:<instanceID> - where <instanceID> is a GUID or UUID
#string id
# Value is always the name of the messageType
#string type
# timestamp (current time) - FIWARE DateTime: Provide "DateTime::" at the beginning
time timestamp
# status message - Human readable description / explanation of the status; default status is "OK"
string status
